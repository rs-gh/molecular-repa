# 20260127
- Set up chemprop encoder with default weights

# 20260121

## Questions to check with TABASCO-REPA
1. Do you still need positional encodings? Does the inductive bias from encoder alignment mean this is no longer necessary?
2. What time steps is alignment most useful for? t ~ 0? t ~ 1?
2. Training curve for tabasco-mild? Should we compare against that?
3. What encoder should we use? MACE? ChemProp?
4. What is the minimal size useful test we can run? Where?

## [REPA](https://arxiv.org/pdf/2410.06940)

We demonstrate that the training
process for generative diffusion models becomes significantly easier and more effective when sup-
ported by an external representation, y∗. Specifically, we propose a simple regularization technique
that leverages recent advances in self-supervised visual representations as y∗, leading to substantial
improvements in both training efficiency and the generation quality of diffusion transformers.
- one key difference for us will be we don't use SSL based representations, but rather some other form of encoder.

Similar to prior studies (Xiang et al., 2023), we first observe that pretrained diffusion models
do indeed learn meaningful discriminative representations (as shown by the linear probing results in
Figure 2a). However, these representations are significantly inferior to those produced by DINOv2.
Next, we find that the alignment between the representations learned by the diffusion model and
those of DINOv2 (Figure 2b) is still considered weak,1 which we study by measuring their repre-
sentation alignment (Huh et al., 2024). Finally, we observe this alignment between diffusion models
and DINOv2 improves consistently with longer training and larger models (Figure 2c).

These insights inspire us to enhance generative models by incorporating external self-supervised rep-
resentations. However, this approach is not straightforward when using off-the-shelf self-supervised
visual encoders (e.g., by fine-tuning an encoder for generation tasks). The first challenge is an input
mismatch: diffusion models work with noisy inputs ˜x, whereas most self-supervised learning en-
coders are trained on clean images x. This issue is even more pronounced in modern latent diffusion
models, which take a compressed latent image z = E(x) from a pretrained VAE encoder (Rombach
et al., 2022) as input. Additionally, these off-the-shelf vision encoders are not designed for tasks like
reconstruction or generation. To overcome these technical hurdles, we guide the feature learning of
diffusion models using a regularization technique that distills pretrained self-supervised representa-
tions into diffusion representations, offering a flexible way to integrate high-quality representations.

Figure 3
- a. unlike with baseline where the semantic gap i.e performance on ImageNet classification gets smaller for larger layer index, with REPA, it looks like the gap starts small and gets bigger somehow

Notably, with improved
alignment, we can push the SiT model’s generation-representation envelope: within the same num-
ber of training iterations, it delivers both better generation quality and stronger linear probing results.
We use a single network trained with REPA at layer 8 and perform the evaluation at different layers.

In essence, REPA dis-
tills the pretrained self-supervised visual representation y∗ of a clean image x into the diffusion
transformer representation h of a noisy input ˜x. This regularization reduces the semantic gap in the
representation h (Figure 3a) and better aligns it with the target self-supervised representations y∗
(Figure 3b). Notably, this enhanced alignment significantly boosts the generation performance of
diffusion transformers (Figure 3c). Interestingly, with REPA, we observe that sufficient represen-
tation alignment can be achieved by aligning only the first few transformer blocks. This, in turn,
allows the later layers of the diffusion transformers to focus on capturing high-frequency details
based on the aligned representations, further improving generation performance

### Setup: Standard Diffusion Transformer + Frozen Encoder

```
model = DiffusionTransformer()
teacher_enc = FrozenDINOv2() # Pretrained & Frozen
projector = MLP(dim_student, dim_teacher) # Small trainable layer
lambda_repa = 0.5 # Weight of the regularization

def training_step(batch_images):
    # 1. Get Teacher representation of clean images
    with torch.no_grad():
        target_repr = teacher_enc(batch_images) # e.g., [B, N, 1024]

    # 2. Standard Diffusion setup
    noise = torch.randn_like(batch_images)
    t = torch.randint(0, 1000, (batch_images.shape[0],))
    noisy_images = add_noise(batch_images, noise, t)

    # 3. Forward pass with hidden state extraction
    # We ask the model to return the features from Layer 6
    pred_noise, hidden_states = model(noisy_images, t, return_layer=6)

    # 4. Project student features to teacher space
    projected_repr = projector(hidden_states)

    # 5. Calculate Losses
    loss_denoise = mse_loss(pred_noise, noise)
    loss_repa = mse_loss(projected_repr, target_repr)

    total_loss = loss_denoise + (lambda_repa * loss_repa)

    # 6. Backprop
    total_loss.backward()
    optimizer.step()
```

# 20251221

## Diffusion

General idea
1. Start with a dataset
2. At each time step $t$, add Gaussian noise -- as time goes to infinity this will yield a fully Gaussian noise distribution
3. Then learn a NN to revert one of those noise-inducing steps


## Flow matching

### [Flow Matching Notes and Code](https://scontent-lhr8-1.xx.fbcdn.net/v/t39.2365-6/469963300_2320719918292896_7950025307614718519_n.pdf?_nc_cat=108&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=TKEh9-YHJb4Q7kNvwG0Zxfx&_nc_oc=AdmYWrXsXw4J5KkCV3-cafWYG3vYgtEIad0ThotckIyC4GB9bEtMB_Qq5iCEQewsWdc&_nc_zt=14&_nc_ht=scontent-lhr8-1.xx&_nc_gid=zAyIXhQKL6-4zj2izSnQjA&oh=00_Afk5dKkC3W3nYunaQQ8QwQOudg9Sd7o18ecImyfd6SHWbQ&oe=694E3C02)

See notebook pages 49-55 and Gemini [chat](https://gemini.google.com/app/dca2f029d12af002)

### [Yannic Kilcher video](https://www.youtube.com/watch?v=7NNxK3CqaDk)

Idea: instead of doing the iterative noise adding/removal process in diffusion, why not just work on the distributions directly? Can we just morph the original noise distribution into the data distribution? Without explicity specifying the process as diffusion does

"How do you learn a flow from one distribution to another while only having samples from the second?"

Paper suggests that solving this problem for individual/single paths and aggregating them is enough to characterise the entire distribution


Flow matching is a more general version of diffusion

This paper is mostly about 'how do you construct distributions from samples'
1. Construct a target prob dist using the samples we have -- for each sample, create a Gaussian with mean = sample, then a Gaussian mixture model across all individual Gaussians?
2.

### [Outlier video](https://www.youtube.com/watch?v=7cMzfkWFWhI)

This video deals with samples. not probability distributions

- Assume that the underlying data distribution that data x1 comes from is $p_1(x_1)$
- Want to learn a NN that helps you approximate and sample from this unknown
- Idea: Start from some known $p_0(x_0)$ and convert it into  $p_1(x_1)$
  - Gaussian noise -> data distribution, following some path
  - Want to learn some NN that learns this path
- Easiest way to construct the path is linear interpolation $x_t = tx_1 + (1-t)x_0$
- This is the optimal transport (OT) displacement map -- the 'straightest' possible path between $x_0$ and $x_1$ i.e. between noise and data
- Want to know dxt/dt because this tells you how much xt changes and therefore how to move closer to data
- $\frac{d}{dt}x_t = x_1 - x_0$  = vector that points towards x1 from x0
- So you want to learn this!

- Different ways to approach learning $p_t(x_t)$:
  - Score matching: use $\nabla_{x_t} \log p_t(x_t)$
  - DDPM: start from $ -\log p_t(x_t)$
  - Flow matching: $p_t(x_t)$ is generated by a ‘flow’
    - $x_t = \psi_t(x)$
      - so $x_t$ can be re-created from some original point $x$ using the flow which is time-indexed
    - $\frac{d}{dt}\psi_t(x) = u_t(\psi_t(x))$
    - this is the same as saying $\frac{d}{dt}x_t = u_t(x_t)$
    - $u_t$ is a vector field -- points in the direction to move towards the data distribution
      - Note that the vector field is time-dependent i.e. it changes with $t$ as well
      - However the optimal transport parameterisation skips this by just assuming time-dependency because the vector field is constant $(x_1 - x_0)$ for all $t$
    - flow matching tries to approximate this flow vector field $u_t$ using a learned NN $v_t$, so the outputs of both functions looks the same given $x_t$
- Objective?
  - want to learn a NN that learns an approximation of the derivative of the flow with respect to time
  - During training:
    - sample random noise $x_0$
    - sample data $x_1$ from dataset
    - sample $t$ between 0 and 1
    - calculate $x_t$ by linearly interpolating between $x_0$ and $x_1$ with the flow = $t x_1 + (1-t) x_0$
    - this is input to NN
    - NN tries to predict vector from $x_0$ to $x_t$
    - input = ($x_t$, $t$) -> output = $v_t(x_t)$
    - minimize $|| v_t(x_t) - (x_1 - x_0) ||^2$
    - this is the *conditional flow matching objective*
      - in reality, we don't know $u_t$ for the entire distribution
      - but we do know the vector field for a single pair of $(x_0, x_1)$
      - The key idea is that by sampling many pairs, the network attmepts to learn the 'average' vector field that transforms the whole noise distribution into the whole data distribution
- Sampling?
  - Numerical integration?
  - Start with $x_0$ = random noise
  - set number of steps e.g. 1000
  ```
  x_t = random(n, d)
  for t in range(steps):
    x_t = x_t + (1/steps) * model(x_t, t)
  ```
- Why doesn't flow matching collapse to a single solution?
  - We always use $(x_1 - x_0)$ as the target, so why doesn't the model just always output this constant?
  - The answer is that the model never sees $x_1$ or $x_0$ -- it only sees their mixture $x_t$. But $x_t$ could lie on a line between any two points $x_0$ and $x_1$!
  - So the model never knows 'Which $x_1$ should I solve for?'
- Why not just use $x_1$ as the target instead of $(x_1 - x_0)$?
  - $x_1$ as target would turn this into an auto-encoder or de-noiser?
  - Given we start from random noise, and there are many possible $x_1$s, using $x_1$ as a target would cause the model to always predict the mean $x_1$ as the output --> model collapse?
  - The flow matching + REPA project kind of does this?
    - flow matching = how to move (target is $(x_1 - x_0)$)
    - REPA = what are you moving towards (target is $x_1$)
- REPA and flow matching
  - The Flow Matching view: The model is trying to find the "average" vector field vt​(xt​,t). To do this well, it needs to understand the "meaning" of xt​.
  - The REPA Intervention: You are adding a loss term that says: "While you are calculating the velocity at xt​, your internal hidden layers should produce a feature vector that is similar to what MACE would produce for a molecule at that same stage."
  - Addressing the "Noisy Molecule" Problem
    - You asked earlier if the vector field is time-independent. In your thesis, t is actually your best friend.
    - At t=1, xt​ is a clean molecule. MACE or Chemprop will give you a very clear, meaningful embedding.
    - At t=0, xt​ is Gaussian noise. A molecular encoder like MACE might give you "garbage" embeddings because it wasn't trained on random noise.
    - Your Challenge: You will need to see if aligning representations at t≈0 (high noise) is actually helpful, or if REPA only provides a useful signal when the "molecule" starts to take shape (t>0.5).
    ```# Training Loop
    for x1 in dataloader: # x1 is a real molecule (Batch, D)
        x0 = torch.randn_like(x1) # x0 is noise (Batch, D) - SAME DIMENSION
        t = torch.rand(1) # random time

        # 1. Create the point on the path (Linear Interpolation)
        xt = (1 - t) * x0 + t * x1

        # 2. Get the model's predicted velocity AND its internal features
        # v_pred: (Batch, D), features: (Batch, Hidden_Dim)
        v_pred, features = model(xt, t)

        # 3. Flow Matching Loss: Follow the straight line
        target_v = x1 - x0
        loss_fm = F.mse_loss(v_pred, target_v)

        # 4. REPA Loss: Align with pre-trained encoder (e.g., MACE)
        with torch.no_grad():
            target_features = pretrained_encoder(x1) # Align with clean data representation
        loss_repa = F.mse_loss(features, target_features)

        # Total Loss
        total_loss = loss_fm + lambda_repa * loss_repa
        total_loss.backward()
    ```
  - Flow Matching works best when the data x1​ shares an underlying structure. In molecules, that structure is "Physics and Valency." As long as that structure exists, the model can handle immense diversity (millions of different molecules) because the "rules" for moving atoms into valid positions are consistent across the whole set.

# 20251116

Running questions
- What is flow matching and how does it work?
Training?
Sampling?
- What is a flow based transformer?
- How is a flow connected to a diffusion model?
- What is equivariance? Vs non-equivariance?
- What is a small molecule vs a protein?
- What are the various facets/properties we care about in the protein modelling task?
- Do different models optimize for different facets?
- Is the idea to combine them together through something like REPA?
- What are the different data sources? What do they all contain?
UniRef?
PDB?
AFDB?
- What is single sequence vs MSA?
- Does PDB and AFDB have either or both?

## [Proteina](https://arxiv.org/pdf/2503.00710)

What is it?
- ‘Flow based protein backbone generator’
- Uses hierarchical fold class labels for conditioning, relies on a tailored scalable transformer architecture with 5x as many params as previous models
- New metrics to measure performance to measure the distributional similarity of generated proteins with reference sets
- Scale training data to millions of synthetic protein structures
- Better training + sampling recipes adapted for protein backbone generation
  - LoRA for protein backbones
  - Classifier free guidance
  - Autoguidance
  - New training objectives
- SOTA on de novo protein backbone design, produces diverse and designable proteins up to 800 residues
- Hierarchical conditioning offers control – enabling high level secondary-structure guidance as well as low-level fold-specific generation

Introduction
- Protein structure -> protein function
- So an approach to de novo protein design is: model the distribution of 3D protein structures, typically with diffusion or flow based methods
- Usually synthesize backbones only
  - unlike PLMs, which often model sequences instead
  - And unlike sequence-to-structure folding models like AlphaFold
Contemporary efforts
Small training data: 500k structures
Neural networks don’t offer any control during synthesis and are usually small
Question: can we scale and control protein structure diffusion and flow models?

Contribution
- Scale protein structure generation and develop a new flow matching-based protein backbone generative model
- In vision and language, generative models are typically prompted through semantic text or class inputs, offering enhanced controllability
- Similarly, enrich the training data with hierarchical fold class labels following the CATH protein structure classification scheme
- This offers both high-level control – eg over secondary structure content
- And low-level guidance wrt specific fold classes – which can eg increase the number of \beta sheets in generated proteins
- Also scale the training data – train on 21MM protein structures
  - Q: from where?
- New scalable xformer architecture
  - Non equivariant design, based on recent vision diffusion transformers
  - Include triangle layers
  - Q: What is this?
- Show that non-equivariant flow models also succeed on unconditional protein structure generation
  - SOTA is equivariant methods
- 400MM params, 5x larger than RFDiffusion
- New metrics for performance, to measure the distribution, not just individual samples
  - SOTA criteria that matter for a protein generator
    - Diversity
    - Novelty
    - Designability
  - Problem: none of these metrics scores models at the distribution level, although the task of generative modeling is to learn the model of a data distribution
  - New metrics introduced to directly score the learned distribution instead of individual samples
    - Similar to the Frechet inception distance, compare sets of generated samples against reference distributions in a non-linear feature space
    - Since the feature extractor is based on a fold class predictor, quantify models diversity over fold classes as well the as similarity of the generated class distribution compared to reference data’s classes
- Adjust the flow matching objective to protein structure generation and explore stage wise training strategies
  - Using LoRA, fine-tune Proteina models on natural, designable proteins
  - New guidance schemes for hierarchical fold class conditioning
- Some discussion of how flow matching works – idk what this stuff is

Data sources
- PDB
  - If you filter this to natural proteins, gives you about 20k
- AFDB
  - Total number here = 214MM
  - But most are not useful for training protein structure generators, mostly contain low quality predictions and other unsuitable data
  - Foldseek AFDB = based on sequential filtering and clustering of the AFDB with the sequence based MMseqs2 and the structure based Foldseek
    - This data uses cluster representatives only i.e. one structure per cluster
    - For the training dataset, use between 32 and 256 residues in the models ~600K in total
      - Q: why not longer or shorter?
    - High quality filtered AFDB dataset
      - 21MM structures
- Synthetic alphafold2 structures
- Question: Does scaling protein structure training data improve model performance in the same way it did for images and NLP?

Can we include further info to the model?
- Eg for images typically rely on semantic class or text-conditioning to offer control or to break down the generative modelling task into a set of simpler conditional tasks
- What does this look like in protein land?
- ‘Hierarchical fold class annotations’ from TED = The Encyclopedia of Domains
  - ‘Structural domain assignments to proteins in the AFDB’
  - CATH hierarchy
    - C = class = overall secondary structure of a domain
      - Alpha helix
      - Beta sheet
      - mixed
    - A = architecture = groups together domains with high structural similarity
    - T = topology/fold = further refines the structure groupings
    - H = homologous superfamily = shared between domains with evolutionary relationships
  - Discard H, use only CAT, since we are only interested in structural modelling
- Assign these labels to the proteins in all datasets

Training objective
- Model protein backbones’ residue locations through their alpha carbon atom coordinates
  - Many works instead leverage ‘frames’ to additionally capture residue rotations
  - But this introduces a whole class of geometry problems

## [Representation Alignment for Generation](https://arxiv.org/pdf/2410.06940)

What is the idea?
- The denoising process in generative diffusion models can induce meaningful discriminative representations inside the model
- But the quality of these representations lags behind self supervised methods
- Argument: one main bottleneck in training large-scale diffusion models for generation lies in effectively learning these representations
- Moreover, training can be made easier by incorporating high-quality external visual representations, instead of relying on the diffusion models to learn them independently

Diffusion models as representation learners
- They learn discriminative features in their hidden states
- Better diffusion models learn better representations
- Denoising score matching as an SSL method?
  - Implicitly learns representation h as hidden state of a denoising autoencoder through a reconstruction of x from corrupt data x_hat
  - But is reconstruction a suitable task for learning good representations?
    - It is not capable of eliminating unnecessary details in x for representation learning
    - Q: ??

This paper
- Identify: the main challenge in training diffusion models stems from the need to learn high quality h
- Demonstrate: Training process for generative diffusion models becomes easier and more effective when supported by an external representation y_star
- How? Use a simple regularization technique to use an SSL representation as y_star
- Improves training efficiency and generation quality
  - Aligns patch wise projections of the model’s hidden states with pretrained SSL representations
  - Clean image = target
  - Goal of the regularization is for the diffusion transformer’s hidden states to predict noise-invariant, clean visual representations from noise inputs that contain useful semantic information
  - This provides meaningful guidance for subsequent layers to reconstruct the target

## [ESM2](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1.full.pdf)

What is it?
- 15B param language model
- Idea: As models scale, they learn information enabling the prediction of the 3D structure of a protein at the resolution of individual atoms
- Inference is order of magnitude faster than AlphaFold2

Idea: because the structure and function of a protein constrains the mutations to its sequence that are selected through evolution, it should also be possible to infer biological structure from sequence patterns
- Structure determines function
- Function determines sequence
- So we should be able to go the other way?
- Predict structure from sequence!

We posit that the task of filling in missing amino acids in
protein sequences across evolution will require a language
model to learn something about the underlying structure
that creates the patterns in the sequences.

Protein language model = train a model with a simple language modelling objective purely on protein sequence data

Argument: large protein language models learn sufficient information to enable accurate, atomic-level predictions of protein structure
ESM2 = language model
ESMFold = uses ESM2 information and representations to perform end-to-end 3D structure prediction using only a single sequence as input
- Can you quantify the emergence of protein structure as the model scales from millions to billions of params?
- Q: What is MSA vs single sequence?

Data
 - UniRef database
Training
- Given an input protein, 15% of amino acids are masked
- ESM2 tasked with predicting these missing positions


What is the evolutionary scale??

Really simple model?
