This work tackles a novel regularization scheme of Large Language Models (LLM) to shape the attention and internal representation of the model during finetuning such that extracted explanations for multiple (similar) decision possibilities are maximally contrastive. Explanations are given by the highest attributed Tokens of the input, extract by Integrated Gradients. This approach ideally leads to an attention shift of the model to the unique Tokens of the input to be perceived as relevant, making explanations less ambiguous. To incorporate the learning for contrastive attributions during model finetuning, the loss function $\mathcal{L}(x,y;f_\theta)$ is supplemented with regularization terms that measure the degree of attribution contrastivity. For multiclass classification problems, the classification loss is evaluated using the Cross Entropy loss function: 
<p align="center">
  $\mathcal{L}_{class} = -\log\frac{e^{z_{y}}}{\sum_{c=1}^{|\mathbb{C}|}e^{z_c}}$ </p>
For optimizing the attributions to be contrastive, a triplet loss is utilized that directly operates on the attributions of positive and negative counterexamples, which are artificially constructed via perturbations in the embedding space:
<p align="center">
  <br/>
  $\mathcal{L}_{triplet} = \max(0,d[A(x,y,f_\theta),A(\smash{x}',y,f_\theta)]-d[A(x,y,f_\theta),A(\smash{x}'',\smash{\hat{y}}'',f_\theta)]+\alpha)$<br/>
  $where$<br/>
  $\smash{x}'\sim N(x,\epsilon),~\smash{x}''\not\in N$ </p>

The attribution scores are given by $A(\cdot)$ and are extracted using Integrated Gradients. The distance function $d(\cdot)$ is used to compare the attributions and is chosen to be the cosine distance, which is the complement of cosine similarity:
<p align="center">
  $cos_{dist} = 1-cos_{sim}$<br/>
  $cos_{sim}(\boldsymbol{a},\boldsymbol{b}) = \frac{\boldsymbol{a}\cdot \boldsymbol{b}}{{\lVert\boldsymbol{a}\rVert_2\lVert\boldsymbol{b}\rVert_2}}$</p>

The positive example $x'$ and the negative example $x''$ are constructed via perturbations in the embedding space $\mathcal{E}$. The perturbation of the positive example is constrained by an $\epsilon$-neighborhood $N$, i.e. a vector is drawn from inside an $\epsilon$-ball around the embedded vector of input $x$. The negative example is constrained to lie outside the $\epsilon$-neighborhood, ensuring a minimum distance to the input and the positive example. For the perturbed negative example to be not arbitrarily far away, it is constrained by a second radius $\epsilon_2$, forming a multidimensional annulus.
<p align="center">
<img src="https://github.com/user-attachments/assets/c985f040-74ce-43ed-b7d1-3a47cd400499" width="400"/>
</p>
The model should learn that the positive example belongs to the current input's class, while the negative example should belong to another (similar) class.
<p align="center"><br/>
  $\mathop{\mathrm{argmax}}_{i \in \{1,\dots,\left|\mathbb{C}\right|\}} f_\theta(x) = \mathop{\mathrm{argmax}}_{i \in \{1,\dots,\left|\mathbb{C}\right|\}} f_\theta(x')$<br/>
  $\mathop{\mathrm{argmax}}_{i \in \{1,\dots,\left|\mathbb{C}\right|\}} f_\theta(x) \neq \mathop{\mathrm{argmax}}_{i \in \{1,\dots,\left|\mathbb{C}\right|\}} f_\theta(x'')$</p>

To formulate both constraints as a loss function, the probability distribution of the model's prediction is utilized: The classification head $f^c_\theta$ consists of a linear projection of the model's representation $r$ into output logits $z$ and a Softmax function, that transforms the logits into a probability distribution over the classes $\mathbb{C}$. The prediction $\hat{y}$ is then given by the largest class probability. The divergence of the probability distributions $P_{x'}$ and $P_{x''}$ to the probability distribution of the current input $P_{x}$ can then be measured using the Jensen-Shannon-Divergence (JSD):
<p align="center">
  JSD(P || Q) = $\frac{1}{2}$ $\text{D}_{\text{KL}}$(P || M) + $\frac{1}{2}$ $\text{D}_{\text{KL}}$(Q || M) <br/>
  $M = \frac{1}{2}(P+Q)$</p>

where $M$ is a mixture distribution of both probability distributions $P$ and $Q$, and $D_{\text{KL}}$ is the Kullback-Leibler Divergence. The divergence between the probability distribution $P_x$ and the positive example $P_{x'}$ should be minimal and can therefore be utilized as a loss function as follows:
<p align="center">
$\mathcal{L}_{\text{positive}} = \text{JSD}(P_x \parallel P_{x'})$</p>

To incorporate the constraint for the negative example $x''$ the JS-Divergence between $P_{x''}$ and a custom uniform distribution $U_{custom}$ is minimized. This is done because it should be prohibited for the model to learn the perturbations of the negative examples to belong to only one class (clustering the noise pattern). For every input $x$ with label $y$ a uniform distribution $U_{custom}$ is constructed where the probability of the current class $y$ is set to $0$ while the probability for all other classes is uniformly set to $\frac{1}{\left|\mathbb{C}\right|-1}$. This principle encourages the model to lower the prediction probability of the input class $y$ while preventing excessive probability mass from being assigned to a single alternative class across all negative examples. The loss function can be expressed as follows:
<p align="center">
  $\mathcal{L}_{negative} = \text{JSD} (P_{x''}\left|\right|U_{custom})$</p>

Finally, the overall loss function is the sum of all loss terms. The optimal model $f^*_{\theta}$ is the model whose learned weights minimize this loss:
<p align="center">
  $\mathcal{L}_{total} = \mathcal{L}_{class}+\mathcal{L}_{triplet}+\mathcal{L}_{positive}+\mathcal{L}_{negative}$<br/><br/>
  $f^*_{\theta} = \mathop{\mathrm{argmin}}_{\theta}~\mathbb{E}_{(x,y)\sim p_{data}}[\mathcal{L}_{total}(x,y,f_\theta)]$

## Example

<p align="center">
<img src="https://github.com/user-attachments/assets/5df23797-2e9c-4e47-8424-8aa7968024f5" width="600"/>
</p>

This example text comes from the [20newsgroups dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html), which consists of user comments of 20 different news topics, e.g. sports-related such as 'baseball' or 'hockey'. A [BERT](https://huggingface.co/docs/transformers/model_doc/bert) language model is finetuned using the above optimization scheme. The first output shows the model's top influenced tokens for the decision without any regularization. The second output shows the result of the finetuned model, where not only the attributions (A) but also the CLS embeddings of the inputs (B) are aligned and contrasted during finetuning. In this example, the model indeed focussed less on ambiguous Tokens (such as game, games, team) and more on unique, class-specific Tokens such as 'Phillies' and 'innings'.
