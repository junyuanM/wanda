from transformers.trainer import Trainer 
import torch 
import torch.nn as nn 
#在模型的层次结构中查找指定类型的层，接受一个模型（通常是nn.Module对象）和一个包含要查找的层类型的列表作为输入，并返回一个字典，其中包含了模型中找到的每个指定类型的层
def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
#修复梯度中出现的NaN和inf值
def fix_grad_nan_inf(model):
    layers = model.model.layers
    count = 0 
    total_params = 0
    for m in model.parameters():
    	if m.requires_grad:
    		if torch.isnan(m.grad).any() or torch.isinf(m.grad).any():
    			m.grad.zero_()

#将梯度中的部分参数置零，以实现梯度掩码的效果。它接受一个模型作为输入，并遍历模型的所有层，将指定条件下的梯度置零
def mask_grad(model):
    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            mask = (W==0)
            subset[name].weight.grad[mask]= 0
 #检查模型中权重的稀疏度。它接受一个模型作为输入，并返回模型中所有权重中零值的比例
def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        # print(f"layer {i} sparsity {float(sub_count)/sub_params:.4f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 
#继承自Trainer类，并覆盖了其中的training_step和compute_loss方法。在training_step方法中，通过调用mask_grad函数实现了梯度掩码的功能，即将梯度中的部分参数置零；
#在compute_loss方法中，添加了处理label smoothing的逻辑，并根据模型类型和返回结果的格式选择合适的处理方式
class SparseTrainer(Trainer):
    def __init__(self, model= None, args= None, data_collator= None, train_dataset= None, eval_dataset= None, 
            tokenizer= None, model_init= None, compute_metrics= None, callbacks= None, optimizers= (None, None),
            preprocess_logits_for_metrics= None
            ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, 
                                optimizers, preprocess_logits_for_metrics)
        self.counter = 0

    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.

        access optimizer through: self.optimizer.optimizer.param_groups[0] 
        """
        self.counter += 1
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling: ## False 
            self.scaler.scale(loss).backward()
        elif self.use_apex:   ## False 
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
            # pass 

        mask_grad(model)   ### mask the gradients

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.

        ## model type: transformers.models.llama.modeling_llama.LlamaForCausalLM
        ## outputs[0]: a single scalar
        ## outputs[1]: shape (bs, 2048, 32000)

        ## inputs["input_ids"] shape: (bs, 2048)
        ## inputs["attention_mask] shape: (bs, 2048)
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
