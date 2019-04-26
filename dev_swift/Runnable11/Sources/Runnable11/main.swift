// export
import Path
import TensorFlow
import nvToolsExt

let path = downloadImagette()

let il = ItemList(fromFolder: path, extensions: ["jpeg", "jpg"])

let sd = SplitData(il, fromFunc: {grandParentSplitter(fName: $0, valid: "val")})

var (procItem,procLabel) = (NoopProcessor<Path>(),CategoryProcessor())

let sld = SplitLabeledData(sd, fromFunc: parentLabeler, procItem: &procItem, procLabel: &procLabel)

let rawData = sld.toDataBunch(itemToTensor: pathsToTensor, labelToTensor: intsToTensor, bs: 32)

let data = transformData(rawData, tfmItem: { openAndResize(fname: $0, size: 128) })

let batch = data.train.oneBatch()!

print(batch.xb.shape)
print(batch.yb.shape)

let labels = batch.yb.scalars.map { procLabel.vocab![Int($0)] }
// showImages(batch.xb, labels: labels)

public struct ConvLayer: FALayer {
    public var bn: FABatchNorm<Float>
    public var conv: FANoBiasConv2D<Float>
    @noDerivative public var delegate: LayerDelegate<Output> = LayerDelegate()
    
    public init(_ cIn: Int, _ cOut: Int, ks: Int = 3, stride: Int = 1, zeroBn: Bool = false, act: Bool = true){
        bn = FABatchNorm(featureCount: cOut)
        if act {
          conv = FANoBiasConv2D(cIn, cOut, ks: ks, stride: stride, activation: relu)
        } else {
          conv = FANoBiasConv2D(cIn, cOut, ks: ks, stride: stride, activation: identity)
        }
        if zeroBn { bn.scale = Tensor(zeros: [cOut]) }
    }
    
    @differentiable
    public func forward(_ input: TF) -> TF {
        return bn(conv(input))
    }
}

//A layer that you can switch off to do the identity instead
public protocol SwitchableLayer: Layer {
    associatedtype Input
    var isOn: Bool {get set}
    
    @differentiable func forward(_ input: Input) -> Input
}

public extension SwitchableLayer {
    func call(_ input: Input) -> Input {
        return isOn ? forward(input) : input
    }

    @differentiating(call)
    func gradForward(_ input: Input) ->
        (value: Input, pullback: (Self.Input.CotangentVector) ->
            (Self.CotangentVector, Self.Input.CotangentVector)) {
        if isOn { return valueWithPullback(at: input) { $0.forward($1) } }
        else { return (input, {v in return (Self.CotangentVector.zero, v)}) }
    }
}

public struct MaybeAvgPool2D: SwitchableLayer {
    var pool: FAAvgPool2D<Float>
    @noDerivative public var isOn = false
    
    @differentiable public func forward(_ input: TF) -> TF { return pool(input) }
    
    public init(_ sz: Int) {
        isOn = (sz > 1)
        pool = FAAvgPool2D<Float>(sz)
    }
}

public struct MaybeConv: SwitchableLayer {
    var conv: ConvLayer
    @noDerivative public var isOn = false
    
    @differentiable public func forward(_ input: TF) -> TF { return conv(input) }
    
    public init(_ cIn: Int, _ cOut: Int) {
        isOn = (cIn > 1) || (cOut > 1)
        conv = ConvLayer(cIn, cOut, ks: 1, act: false)
    }
}

public struct ResBlock: FALayer {
    @noDerivative public var delegate: LayerDelegate<Output> = LayerDelegate()
    public var convs: [ConvLayer]
    public var idConv: MaybeConv
    public var pool: MaybeAvgPool2D
    
    public init(_ expansion: Int, _ ni: Int, _ nh: Int, stride: Int = 1){
        let (nf, nin) = (nh*expansion,ni*expansion)
        convs = [ConvLayer(nin, nh, ks: 1)]
        convs += (expansion==1) ? [
            ConvLayer(nh, nf, ks: 3, stride: stride, zeroBn: true, act: false)
        ] : [
            ConvLayer(nh, nh, ks: 3, stride: stride),
            ConvLayer(nh, nf, ks: 1, zeroBn: true, act: false)
        ]
        idConv = nin==nf ? MaybeConv(1,1) : MaybeConv(nin, nf)
        pool = MaybeAvgPool2D(stride)
    }
    
    @differentiable
    public func forward(_ inp: TF) -> TF {
        return relu(convs(inp) + idConv(pool(inp)))
    }
    
}

func makeLayer(_ expansion: Int, _ ni: Int, _ nf: Int, _ nBlocks: Int, stride: Int) -> [ResBlock] {
    return Array(0..<nBlocks).map { ResBlock(expansion, $0==0 ? ni : nf, nf, stride: $0==0 ? stride : 1) }
}

public struct XResNet: FALayer {
    @noDerivative public var delegate: LayerDelegate<Output> = LayerDelegate()
    public var stem: [ConvLayer]
    public var maxPool = MaxPool2D<Float>(poolSize: (3,3), strides: (2,2), padding: .same)
    public var blocks: [ResBlock]
    public var pool = GlobalAvgPool2D<Float>()
    public var linear: Dense<Float>
    
    public init(_ expansion: Int, _ layers: [Int], cIn: Int = 3, cOut: Int = 1000){
        var nfs = [cIn, (cIn+1)*8, 64, 64]
        stem = Array(0..<3).map{ ConvLayer(nfs[$0], nfs[$0+1], stride: $0==0 ? 2 : 1)}
        nfs = [64/expansion,64,128,256,512]
        blocks = Array(layers.enumerated()).map { (i,l) in 
            return makeLayer(expansion, nfs[i], nfs[i+1], l, stride: i==0 ? 1 : 2)
        }.reduce([], +)
        linear = Dense(inputSize: nfs.last!*expansion, outputSize: cOut)
    }
    
    @differentiable
    public func forward(_ inp: TF) -> TF {
        return linear(pool(blocks(maxPool(stem(inp)))))
    }
    
}

// export
extension Learner where Opt.Scalar: BinaryFloatingPoint {
    public class StopEarly: Delegate {
        private var numIter: Int
        public init(numIter: Int = 100) { self.numIter = numIter }
        override public func batchDidFinish(learner: Learner) throws {
            if learner.currentIter >= numIter { throw LearnerAction.stop }
        }
    }
    
    public func makeStopEarly(numIter: Int) -> StopEarly {
        return StopEarly(numIter: numIter)
    }
}

/// Adam optimizer.
///
/// Reference: ["Adam - A Method for Stochastic Optimization"](
/// https://arxiv.org/abs/1412.6980v8)
public class FAAdam<Model: Layer>: Optimizer
    where Model.AllDifferentiableVariables == Model.CotangentVector {
    /// The learning rate.
    public var learningRate: Float
    /// A coefficient used to calculate the first and second moments of
    /// gradients.
    public var beta1: Float
    /// A coefficient used to calculate the first and second moments of
    /// gradients.
    public var beta2: Float
    /// A small scalar added to the denominator to improve numerical stability.
    public var epsilon: Float
    /// The weight decay.
    public var decay: Float
    /// The current step.
    public var step: Int = 0
    /// The first moments of the weights.
    public var firstMoments: Model.AllDifferentiableVariables
    /// The second moments of the weights.
    public var secondMoments: Model.AllDifferentiableVariables

    public init(
        for model: __shared Model,
        learningRate: Float = 1e-3,
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        epsilon: Float = 1e-8,
        decay: Float = 0
    ) {
        precondition(learningRate >= 0, "Learning rate must be non-negative")
        precondition(0 <= beta1 && beta1 <= 1, "Beta parameter must be between 0 and 1")
        precondition(0 <= beta2 && beta2 <= 1, "Beta parameter must be between 0 and 1")
        precondition(decay >= 0, "Weight decay must be non-negative")

        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay

        // Initialize first & second moments to be zeros of the same shape.
        // We can't use `Model.AllDifferentiableVariables.zero` due to the
        // interaction between Key Paths and Differentiable Arrays.
        firstMoments = model.allDifferentiableVariables
        secondMoments = model.allDifferentiableVariables
        for kp in firstMoments.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            firstMoments[keyPath: kp] = Tensor<Float>(0)
            secondMoments[keyPath: kp] = Tensor<Float>(0)
        }
    }


    public func update(_ model: inout Model.AllDifferentiableVariables,
                       along direction: Model.AllDifferentiableVariables) {
        step += 1
        let learningRate = self.learningRate * 1 / (1 + decay * Float(step))
        let stepSize = learningRate * (sqrt(1 - pow(beta2, Float(step))) /
            (1 - pow(beta1, Float(step))))

        ApplyAdamUpdate(
            learningRate: Tensor<Float>(learningRate) + Tensor<Float>(0.0),
            beta1: Tensor<Float>(beta1) + Tensor<Float>(0.0),
            beta2: Tensor<Float>(beta2) + Tensor<Float>(0.0),
            not_beta1: Tensor<Float>(1 - beta1) + Tensor<Float>(0.0),
            not_beta2: Tensor<Float>(1 - beta2) + Tensor<Float>(0.0),
            epsilon: Tensor<Float>(epsilon) + Tensor<Float>(0.0),
            stepSize: Tensor<Float>(stepSize) + Tensor<Float>(0.0),
            model: &model,
            direction: direction)
    }

    func ApplyAdamUpdate(
        learningRate: Tensor<Float>,
        beta1: Tensor<Float>,
        beta2: Tensor<Float>,
        not_beta1: Tensor<Float>,
        not_beta2: Tensor<Float>,
        epsilon: Tensor<Float>,
        stepSize: Tensor<Float>,
        model: inout Model.AllDifferentiableVariables,
        direction: Model.AllDifferentiableVariables
     ) {
        // Update Float & Double Tensor variables.
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
nvtxRangePushA("Tensor Update")
            firstMoments[keyPath: kp] =
                firstMoments[keyPath: kp] * beta1 + (not_beta1) * direction[keyPath: kp]
            secondMoments[keyPath: kp] =
                secondMoments[keyPath: kp] * beta2 + (not_beta2) *
                direction[keyPath: kp] * direction[keyPath: kp]
            model[keyPath: kp] -=
                stepSize * firstMoments[keyPath: kp] / (sqrt(secondMoments[keyPath: kp]) + epsilon)
nvtxRangePop()
        }
    }
}


func xresnet18 (cIn: Int = 3, cOut: Int = 1000) -> XResNet { return XResNet(1, [2, 2, 2, 2], cIn: cIn, cOut: cOut) }
func xresnet34 (cIn: Int = 3, cOut: Int = 1000) -> XResNet { return XResNet(1, [3, 4, 6, 3], cIn: cIn, cOut: cOut) }
func xresnet50 (cIn: Int = 3, cOut: Int = 1000) -> XResNet { return XResNet(4, [3, 4, 6, 3], cIn: cIn, cOut: cOut) }
func xresnet101(cIn: Int = 3, cOut: Int = 1000) -> XResNet { return XResNet(4, [3, 4, 23, 3], cIn: cIn, cOut: cOut) }
func xresnet152(cIn: Int = 3, cOut: Int = 1000) -> XResNet { return XResNet(4, [3, 8, 36, 3], cIn: cIn, cOut: cOut) }

func modelInit() -> XResNet { return xresnet50(cOut: 10) }
let optFunc: (XResNet) -> FAAdam<XResNet> = { model in FAAdam(for: model, learningRate: 1e-2, beta1: 0.9, beta2: 0.99, epsilon: 1e-6, decay: 1e-2) }
let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)
let recorder = learner.makeDefaultDelegates(metrics: [accuracy])
learner.addDelegate(learner.makeNormalize(mean: imagenetStats.mean, std: imagenetStats.std))
learner.addDelegate(learner.makeStopEarly(numIter: 10))

nvtxRangePushA("Silly")
try! learner.fit(1)
nvtxRangePop()

// Experiment: Iterate through the whole dataset. This seems to go really fast.
// var xOpt: Tensor<Float>? = nil
// var n: Int = 0
// for batch in data.train.ds {
//   print(n)
//   n += 1
//   guard let x = xOpt else {
//     xOpt = batch.xb
//     continue
//   }
//   guard batch.xb.shape[0] == x.shape[0] else {
//     print("Smaller batch, skipping")
//     continue
//   }
//   xOpt = x + batch.xb
// }
// print(xOpt!)
