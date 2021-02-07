from . import *
from modules import controller, monitor
from . import _layers as layers
from ._metrics import masked_softmax_cross_entropy, masked_accuracy
from datasets import TransformSPAdj
import operator
import inspect
from itertools import cycle
keras = tf.keras


def add_subparser_args(parser):
    subparser = parser.add_argument_group("CPGNN Model Arguments (CPGNN.py)")
    subparser.add_argument("--network_setup", type=str,
                           default="M64-R-MO-E-BP1", help="(default: %(default)s)")
    subparser.add_argument("--dropout", type=float,
                           default=0.5, help="Default dropout rate")
    subparser.add_argument("--hidden", type=int, default=16)
    subparser.add_argument("--adj_nhood", default=["1"], type=str, nargs="+")
    subparser.add_argument("--optimizer", type=str,
                           default="adam", help="(default: %(default)s)")
    subparser.add_argument("--lr", type=float, default=0.01,
                           help="(default: %(default)s)")
    subparser.add_argument("--l2_regularize_weight", type=float, default=5e-4,
                           help="(default: %(default)s)")
    subparser.add_argument("--early_stopping", type=int, default=0,
                           help="Number of epochs used to decide early stopping (default: %(default)s)")
    subparser.add_argument("--save_activations", action="store_true")
    subparser.add_argument("--save_predictions",
                           nargs="+", type=bool, default=True)
    subparser.add_argument("--adj_normalize", choices=[
        "ORDINARY", "SYM_NORMALIZED", "RW_NORMALIZED", "CHEBY"],
        default="SYM_NORMALIZED")
    subparser.add_argument("--best_val_criteria", nargs="+",
                           choices=["val_loss", "val_acc"], default=["val_acc", "val_acc"])
    subparser.add_argument("--train_bp_after", type=int, default=400)
    subparser.add_argument("--no_pretrain", action="store_true")
    subparser.add_argument("--cotrain_weight", type=float, default=1)
    subparser.add_argument("--no_feature_normalize", action="store_true")
    subparser.add_argument("--p_weights", type=float, default=1.0)
    subparser.add_argument("--use_best_val_belief", action="store_true")
    parser.function_hooks["argparse"].append(argparse_callback)


def argparse_callback(args):
    dataset = args.objects["dataset"]
    layer_setups = parse_network_setup(args.network_setup, dataset.num_labels,
                                       _dense_units=args.hidden, _dropout_rate=args.dropout,
                                       parse_preprocessing=True)
    preprocessing_data(args,
                       getAdjNormHops=args.adj_nhood, normType=getattr(TransformSPAdj.NType, args.adj_normalize))
    initialize_model(args, layer_setups, args.optimizer, args.lr,
                     args.l2_regularize_weight, args.early_stopping)


def preprocessing_data(args, **kwargs):
    '''
    Preprocess the data and generate tensors in TensorFlow format.
    '''
    dataset = args.objects["dataset"]
    if not args.no_feature_normalize:
        dataset.row_normalize_features()
    dataset.adj_remove_eye()
    args.objects["tensors"] = vars(
        dataset.getTensors(getDenseAdj=False, **kwargs))
    args.objects["tensors"]["cotrain_weight"] = tf.constant(
        args.cotrain_weight, dtype=tf.float32)


def initialize_model(args, layer_setups, optimizer, lr,
                     l2_regularize_weight, early_stopping, sparse_input=True):

    model = CPGNN(layer_setups, sparse_input=sparse_input,
                   l2_regularize_weight=l2_regularize_weight)
    model.p_weights = args.p_weights
    args.objects["best_val_criteria_iter"] = cycle(args.best_val_criteria)
    args.best_val_criteria = next(args.objects["best_val_criteria_iter"])

    optimizer = keras.optimizers.get(
        optimizer).from_config({"lr": lr})
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    def train_step(adj, adj_hops, features, y_train, train_mask, cotrain_weight, **kwargs):
        addSupervision = (len(model.supervised_inds) > 0)
        with tf.GradientTape() as tape:
            if addSupervision:
                predictions, supervisedOutputs = model(adj, features, adj_hops,
                                                       y_train=y_train,
                                                       train_mask=train_mask,
                                                       training=True,
                                                       addSupervision=True)
            else:
                predictions = model(adj, features, adj_hops,
                                    y_train=y_train,
                                    train_mask=train_mask,
                                    training=True)
            reg_loss = model._loss(predictions, y_train, train_mask)
            pred_loss = masked_softmax_cross_entropy(
                predictions, y_train, train_mask)
            train_loss = reg_loss + pred_loss
            if addSupervision:
                for outputs in supervisedOutputs:
                    train_loss += cotrain_weight * \
                        model._loss(outputs, y_train, train_mask)
        gradients = tape.gradient(train_loss, model.trainable_variables)
        if args.grad_monitor:
            monitor.grad_monitor(model, gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return dict(train_loss=train_loss)

    # @tf.function
    def test_step(adj, adj_hops, features, y_train, train_mask,
                  y_val, val_mask, y_test, test_mask, verbose=args.verbose,
                  save_activations=False, save_predictions=False, **kwargs):
        if args.use_signac:
            if save_activations:
                print("Saving activations to Signac Job Data Storage:")
                predictions = model(adj, features, adj_hops,
                                    y_train=y_train,
                                    train_mask=train_mask,
                                    training=False,
                                    saveActivations=args.objects["signac_job"].data)
                print(args.objects["signac_job"].workspace())
            else:
                predictions = model(adj, features, adj_hops,
                                    y_train=y_train,
                                    train_mask=train_mask,
                                    training=False)

            if save_predictions:
                args.objects["signac_job"].data["predicted_prob"] = toNumpy(
                    predictions)
                for scope, y_scope, scope_mask in (
                    ('train', y_train, train_mask),
                    ('val', y_val, val_mask),
                    ('test', y_test, test_mask)
                ):
                    args.objects["signac_job"].data[f"{scope}_mask"] = toNumpy(
                        scope_mask)
        else:
            predictions = model(adj, features, adj_hops,
                                y_train=y_train,
                                train_mask=train_mask,
                                training=False)
        
        val_loss = masked_softmax_cross_entropy(
            predictions, y_val, val_mask)
        test_loss = masked_softmax_cross_entropy(
            predictions, y_test, test_mask)
        train_acc = masked_accuracy(predictions, y_train, train_mask)
        val_acc = masked_accuracy(predictions, y_val, val_mask)
        test_acc = masked_accuracy(predictions, y_test, test_mask)
        test_stats_dict = dict(train_acc=train_acc, val_acc=val_acc, test_accuracy=test_acc,
                               val_loss=val_loss, test_loss=test_loss, monitor=dict())
        if args.deg_acc_monitor and verbose:
            monitor.deg_acc_monitor(args, args.deg_acc_monitor, adj, predictions,
                                    y_train, train_mask, "train", test_stats_dict["monitor"])
            monitor.deg_acc_monitor(args, args.deg_acc_monitor, adj, predictions,
                                    y_val, val_mask, "val", test_stats_dict["monitor"])
            monitor.deg_acc_monitor(args, args.deg_acc_monitor, adj, predictions,
                                    y_test, test_mask, "test", test_stats_dict["monitor"])
        return test_stats_dict

    def predict_step(adj, adj_hops, features, **kwargs):
        predictions = model(adj, features, adj_hops, training=False)
        return predictions

    def embed_step(adj, adj_hops, features, use_relu=False, **kwargs):
        embeddings = model.getEmbeddings(adj, features, adj_hops)
        return embeddings

    def attn_step(adj, adj_hops, features, **kwargs):
        attnCoeffs = model.getAttnCoeff(adj, features, adj_hops)
        return attnCoeffs

    statsPrinter = logger.EpochStatsPrinter()
    args.objects["statsPrinter"] = statsPrinter
    args.objects["best_val_stats"] = None
    args.objects["current_ckpt"] = None
    args.objects["early_stopping"] = controller.SlidingMeanEarlyStopping(
        early_stopping)

    def post_epoch_callback(epoch, args):
        if epoch == args.train_bp_after + 1:
            model.summary()
        epoch_stats_dict = args.objects["epoch_stats"]
        statsPrinter(epoch, epoch_stats_dict)

        # Early Stopping
        if args.objects["early_stopping"](epoch_stats_dict["val_loss"]):
            print("Early stopping...")
            args.current_epoch = float('inf')

        # Remove previous weights
        current_ckpt = args.objects["current_ckpt"]
        if ((current_ckpt is not None) and (args.objects["best_val_stats"] is not None)
                and (current_ckpt != args.objects["best_val_stats"].get("ckpt"))):
            logger.remove_ckpt(args, current_ckpt)

        # Save Model Weights
        args.objects["current_ckpt"] = logger.save_ckpt(
            checkpoint, args, epoch, epoch_stats_dict)

        # Save Perf Stats for Best Val Acc
        if model.train_bp and args.use_best_val_belief:
            if args.best_val_criteria == "val_acc":
                op = operator.gt
            elif args.best_val_criteria == "val_loss":
                op = operator.lt
        else:
            if args.best_val_criteria == "val_acc":
                op = operator.ge
            elif args.best_val_criteria == "val_loss":
                op = operator.le

        if ((args.objects["best_val_stats"] is None)
                or op(epoch_stats_dict[args.best_val_criteria],
                      args.objects["best_val_stats"][args.best_val_criteria])):
            if args.objects["best_val_stats"] is not None:
                logger.remove_ckpt(
                    args, args.objects["best_val_stats"].get("ckpt"))
            args.objects["best_val_stats"] = dict(epoch_stats_dict)
            args.objects["best_val_stats"]["epoch"] = epoch
            args.objects["best_val_stats"]["ckpt"] = args.objects["current_ckpt"]

        if epoch == args.train_bp_after and not model.train_bp:
            model.train_bp = True
            print("Begin to train LinBP...")
            if args.use_best_val_belief:
                print("Restoring the best performance MLP model")
                logger.restore_ckpt(
                    args.objects["checkpoint"], args, args.objects["best_val_stats"]["ckpt"])
                print("Best performance:")
                statsPrinter.from_dict(args.objects["best_val_stats"])
            model.supervised_inds.append(model.embedding_ind)
            args.best_val_criteria = next(
                args.objects["best_val_criteria_iter"])
            
            # Test pretrain BP performance
            bp_stats = args.objects["test_step"](
                **args.objects["tensors"]
            )
            print("BP performance:")
            statsPrinter(0, bp_stats, train_loss=float('nan'))
            if args.use_signac:
                job = args.objects["signac_job"]
                record_dict = dict()
                for key, item in bp_stats.items():
                    if tf.is_tensor(item):
                        record_dict[key] = item.numpy().item()
                    else:
                        record_dict[key] = item
                with open(job.fn("bp_results.json"), "w") as f:
                    json.dump(record_dict, f)

        elif epoch > args.train_bp_after:
            H_est = model.layer_objs[-1].get_H()
            if args.use_signac:
                job = args.objects["signac_job"]
                job.data.estimated_H[str(epoch)] = toNumpy(H_est)
                if "init_H" not in job.data:
                    try:
                        job.data.init_H = toNumpy(model.layer_objs[-1].H_init)
                    except AttributeError:
                        pass

    def post_train_callback(args):
        if (not args.verbose) or (args.save_activations) or (args.save_predictions):
            print("Restoring the best performance model")
            logger.restore_ckpt(
                args.objects["checkpoint"], args, args.objects["best_val_stats"]["ckpt"])
            epoch_stats_dict = args.objects["test_step"](
                **args.objects["tensors"], verbose=True,
                save_activations=args.save_activations,
                save_predictions=args.save_predictions
            )
            args.objects["best_val_stats"]["monitor"] = epoch_stats_dict["monitor"]
        print("Best performance:")
        statsPrinter.from_dict(args.objects["best_val_stats"])
        if args.use_signac:
            job = args.objects["signac_job"]
            record_dict = dict()
            for key, item in args.objects["best_val_stats"].items():
                if tf.is_tensor(item):
                    record_dict[key] = item.numpy().item()
                else:
                    record_dict[key] = item
            with open(job.fn("results.json"), "w") as f:
                json.dump(record_dict, f)

            if model.H is not None:
                H_est = model.layer_objs[-1].get_H()
                if args.use_signac:
                    job = args.objects["signac_job"]
                    job.data["estimated_H"]["optimum"] = H_est
                    print("Saving compatibility matrix H...")

    args.objects["model"] = model
    args.objects["optimizer"] = optimizer
    args.objects["checkpoint"] = checkpoint
    args.objects["train_step"] = train_step
    args.objects["test_step"] = test_step
    args.objects["predict_step"] = predict_step
    args.objects["embed_step"] = embed_step
    args.objects["attn_step"] = attn_step
    args.objects["post_epoch_callbacks"].append(post_epoch_callback)
    args.objects["post_train_callbacks"].append(post_train_callback)
    if args.use_signac:
        job = args.objects["signac_job"]
        job.data["estimated_H"] = dict()

    if args.no_pretrain:
        model.train_bp = True


class CPGNN(tf.keras.Model):
    def __init__(self, layer_setups,
                 sparse_input=True, l2_regularize_weight=0):
        super().__init__()
        self.layer_objs = []
        self.dropout_inds = []
        self.supervised_inds = []
        self.graph_inds = []
        self.graph_hops_inds = []
        self.concat_inds = []
        self.embedding_ind = None
        self.output_ind = None
        self.tagsDict = dict()

        self.H = None
        self.train_bp = False

        # Hidden Layers
        for ind, (layerType, layerConf) in enumerate(layer_setups):
            layerTag = layerConf.pop("tag", None)
            isEmbedding = layerConf.get("isEmbedding", False)
            if isinstance(layerType, tf.keras.layers.Layer):
                self.layer_objs.append(layerType)
                if hasattr(layerType, "model_regularizer"):
                    layerType.model_regularizer = keras.regularizers.l2(
                        l2_regularize_weight)
                if hasattr(layerType, "sparse_input"):
                    layerType.sparse_input = sparse_input
                    sparse_input = False

            elif layerType == Layer.DENSE:
                beginOutput = layerConf.get("beginOutput", False)
                if beginOutput:
                    self.output_ind = len(self.layer_objs)
                if sparse_input:
                    self.layer_objs.append(layers.SparseDense(
                        layerConf["units"],
                        use_bias=layerConf["use_bias"],
                        kernel_regularizer=keras.regularizers.l2(
                            l2_regularize_weight)
                    ))
                    sparse_input = False
                else:
                    self.layer_objs.append(keras.layers.Dense(
                        layerConf["units"],
                        use_bias=layerConf["use_bias"],
                        kernel_regularizer=keras.regularizers.l2(
                            l2_regularize_weight)
                    ))
            elif layerType == Layer.DROPOUT:
                layerInd = len(self.layer_objs)
                self.dropout_inds.append(layerInd)
                dropout_rate = layerConf["dropout_rate"]
                if sparse_input:
                    self.layer_objs.append(layers.SparseDropout(dropout_rate))
                else:
                    self.layer_objs.append(keras.layers.Dropout(dropout_rate))
            elif layerType == Layer.SLICE:
                self.concat_inds.append(len(self.layer_objs))
                self.layer_objs.append(layers.SliceLayer(
                    **layerConf
                ))
            elif layerType == Layer.IDENTITY:
                self.layer_objs.append(tf.sparse.to_dense)
                sparse_input = False
            elif layerType == Layer.GCN:
                self.graph_hops_inds.append(len(self.layer_objs))
                self.layer_objs.append(layers.GCNLayer(**layerConf))
            elif layerType == Layer.TENSORDOT:
                self.layer_objs.append(layers.TensorDotLayer(
                    layerConf["units"],
                    use_bias=layerConf["use_bias"],
                    kernel_regularizer=keras.regularizers.l2(
                        l2_regularize_weight),
                    sparse_input=sparse_input
                ))
                if sparse_input:
                    sparse_input = False
            elif layerType == Layer.LAMBDA:
                self.layer_objs.append(eval(layerConf["lambda"]))
            elif layerType == Layer.RELU:
                self.layer_objs.append(tf.keras.layers.ReLU())
            elif layerType == Layer.VECTORIZE:
                self.layer_objs.append(tf.keras.layers.Flatten())
            elif layerType == Layer.SUM:
                self.layer_objs.append(layers.SumLayer())
            elif layerType == Layer.CONCAT:
                self.concat_inds.append(len(self.layer_objs))
                self.layer_objs.append(layers.ConcatLayer(
                    tags=layerConf["tags"],
                    addInputs=layerConf["addInputs"]
                ))
            elif layerType == Layer.STOP_GRADIENT:
                self.layer_objs.append(tf.stop_gradient)
            else:
                raise ValueError(
                    f"Unsupported layer type {layerType} specified in this model.")

            if layerConf.get("supervised", False):
                self.supervised_inds.append(len(self.layer_objs) - 1)

            if layerTag:
                if len(self.layer_objs) - 1 in self.tagsDict:
                    print(
                        f"WARNING: overriding tag {layerTag} in layer {len(self.layer_objs) - 1}")
                self.tagsDict[len(self.layer_objs) -
                              1] = layerTag  # ind to Tag

            if isEmbedding:
                self.embedding_ind = len(self.layer_objs) - 1

    def call(self, adj, inputs, adjhops, training=False, returnBefore=0, executeAfter=0,
             addSupervision=False, saveActivations=None, **kwargs):
        supervisedOutputs = []
        taggedOutputs = dict()

        taggedOutputs["adj"] = adj
        taggedOutputs["adjhops"] = adjhops
        taggedOutputs.update(kwargs)

        if saveActivations is not None:
            saveActivations["inputs/inputs"] = toNumpy(inputs)
            saveActivations["inputs/adj"] = toNumpy(adj)
            for i in range(len(adjhops)):
                saveActivations[f"inputs/adjhops/{i}"] = toNumpy(adjhops[i])
        if returnBefore <= 0:
            returnBefore = len(self.layer_objs) + returnBefore
        if not self.train_bp:
            returnBefore = min(returnBefore, self.embedding_ind + 1)
        if executeAfter < 0:
            executeAfter = len(self.layer_objs) + executeAfter
        for ind, layer in enumerate(self.layer_objs):
            if ind == returnBefore:
                return inputs
            elif ind < executeAfter:
                continue

            last_inputs = inputs
            if ind in self.concat_inds:
                inputs = layer(inputs, **taggedOutputs)
            elif hasattr(layer, "ARGS"):
                argDict = {name: taggedOutputs[name] for name in layer.ARGS
                           if name in taggedOutputs}
                argDict["inputs"] = inputs
                inputs = layer(**argDict)
            elif ind in self.graph_hops_inds:
                inputs = layer(adjhops, inputs)
            elif ind in self.graph_inds:
                inputs = layer(adj, inputs)
            else:
                inputs = layer(inputs)

            if addSupervision and (ind in self.supervised_inds):
                supervisedOutputs.append(inputs)

            if saveActivations is not None:
                layer_name = (layer.name if hasattr(
                    layer, "name") else layer.__name__)
                saveActivations[f"activations/{ind}-{layer_name}"] = toNumpy(
                    inputs)

            if ind in self.tagsDict:
                tag = self.tagsDict[ind]
                taggedOutputs[tag] = inputs

        if addSupervision:
            return inputs, supervisedOutputs
        else:
            return inputs

    def callOutputNetwork(self, adj, inputs, adjhops, training=False, returnBefore=0, **kwargs):
        self(adj, inputs, adjhops, training,
             returnBefore, self.output_ind, **kwargs)

    def getEmbeddings(self, adj, inputs, adjhops, **kwargs):
        return self(adj, inputs, adjhops, training=False, returnBefore=self.embedding_ind + 1)

    def getAttnCoeff(self, adj, inputs, adjhops, **kwargs):
        self(adj, inputs, adjhops, training=False)
        attnCoeffList = []
        for ind in self.attention_inds:
            attnCoeffList.append(self.layer_objs[ind].P)
        return attnCoeffList

    def updateH(self, adj, y, inputs=None, sample_mask=None):
        Hinit = layers.CompatibilityLayer.estimateH(
            adj, y, inputs, sample_mask)

    # @tf.function

    def _loss(self, predictions, labels, mask):
        regularization_loss = tf.math.add_n(self.losses)
        return regularization_loss
