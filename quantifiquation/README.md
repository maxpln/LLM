# Quantization

La quantifiquation est une technique qui conssite à réduire les coûts en calcul et en mémoire en lancant une inférence en représentant les poids et les activations avec un type de données avec une précision plus faible.

Réduire le nombre de bits signifie que le modèle résultatnt nécessitera moins de mémoire, conssomera moins d'énergie et les opérations comme les multiplications de matrices pourront être faite plus rapidment.

L'idée de la quantifiquation est de passé d'une représentation haute préciison (32-bit floatting point) pour les poids et les activations à un type de données avec une présion plus faible. Les précision moins précises les plus communes sont:
* float16, accumation data type float16
* bfloat16, accumulation data types float32
* int16, accumulation data type int32
* int8, accumulation data type int32
  
L'accumulation data type spécifie le type du résultat d'une accumulation (addition, multipliquation...). Par exemple, si on a deux int8 valeurs 127 et que l'on veut les sommer. Le résultats est bien plus gros, que la plus grandes valeurs représentables en int8, on va donc avoir besoin d'un type de données avec une précision plus large pour éviter d'avoir une trop grosse perte de précision ce qui rendrait le processus de quantifiquation inutile.

## Quantization methods

Il y a deux méthodes de quantifiquations principales :
* Post-training quantization : méthodes concentré sur la diminution de la précsion après que le modèle est été entrainé. Cependant, ça ne donne pas de meilleurs performances que la quantifiquation pedant l'entrainement.
* Quantization aware training : Cette méthodes permet de quantifier un modèle et ensuite de le finetuné pour réduire les dégardations des peformance due à la quantifiquation. Sinon la quantifiquation peut aussi se dérouler durant l'entrainemnt.

### Quantization to float16

Performing quantization to go from float32 to float16 is quite straightforward since both data types follow the same representation scheme. The questions to ask yourself when quantizing an operation to float16 are:

* Does my operation have a float16 implementation?
* Does my hardware suport float16? For instance, Intel CPUs have been supporting float16 as a storage type, but computation is done after converting to float32.
* Is my operation sensitive to lower precision? For instance the value of epsilon in LayerNorm is usually very small (~ 1e-12), but the smallest representable value in float16 is ~ 6e-5, this can cause NaN issues. The same applies for big values.

### GPTQ Quantization 

GPTQ est une méthode de quantification après l'entrainemnt pour rendre le modèle plus petit avec un dataset de calibration. L'idée derrière GPTQ est simple, ça quantifie chaque poids en trouvant une vesrion compressé de ce poid qui va réduire au maximum une MSE.

Avec GPTQ, on applique une post quantifiquations une fois pour toute, et cela va donner une sauvegarde de mémoire et une accélération de l'inférence (contrairement à la quantifiquation 4/8 bit). AutoGPTQ est une bibliothèque qui permet la quantifiquation GPTQ? 

Les bénéifices de autoGPTQ :
* **rapide pour la génération de texte** : Les modèles quantifié GPTQ sont plus rapides que les modèles auntifiés avec bitsandbytes pour al génération de texte.
* **n_bit support** : l'algo GPTQ rend la quantifiquation possible jusqu'à 2 bits. Cependant, il y un risque de forte dégradations de performances. Le nombre recommendé de bits est 4, qui semble être un bon contreparti pour GPTQ.

Les points d'améliorations possibles :
* L'utilisation d'un dataset de calibration peut décourager certains utilisateurs. De plus, cela peut prendre plusieurs jeures pour quantifier le modèle.
* Fonctionne seulement pour les modèles de langues pour l'instant. 


```
pip install auto-gptq
pip install transformers optimum peft
```
On peut lancer un model GPTQ du hub HuggingFace comme ça :
```
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GPTQ", torch_dtype=torch.float16, device_map="auto")
```
On peut quantifier n'importe quels modèles transfromers comme ça :
```
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)
```

De plus, il est possible de finetuné un modèle GPTQ en utilisant PEFT comme cet [exemple](https://gist.github.com/SunMarc/dcdb499ac16d355a8f265aa497645996). 

###  4/8-bit Quantization with bistandbytes

bistandbytes est une bibliothèque utilisé pour appliquer une quantifiquation 8-bit ou 4-bit. Il peut être utilisé duant l'entrainement pour un entrainement en mixed-precision ou avant l'inférence pour rendre le modèle plus petit. 

bitsandbytes est la façon la plus simple de quantifier un modèle comme ça ne nécessite pas de calibrer le modèle quantifié avec des données en entrée (zero_shot quantization). 

8-bit quantization enables multi-billion parameter scale models to fit in smaller hardware without degrading performance. 8bit quantization works as follows :
1. Extract the larger values (outliers) columnwise from the input hidden states.
2. Perform the matrix multiplication of the outliers in FP16 and the non-outliers in int8.
3. Scale up the non-outlier results to pull the values back to FP16, and add them to outlier results in FP16.

```
pip install bitsandbytes
```
Pour charger un modèle en 8-bit avec transformers :
```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_8bit = AutoModelForCausalLM.from_pretrained(name, device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(name)
encoded_input = tokenizer(text, return_tensors='pt')
output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda())
print(tokenizer.decode(output_sequences[0], skip_special_tokens=True))
```

Un des points noires de la quantifiquation 8-bit avec bitsandbytes et que la vitesse d'inférence est plus lente comparé à GPTQ.

4-bit float (FP4) et 4-bit NormalFloat (NF4) sont deux types de données introduit pour être utilisé avec la technique du QLoRA, pour faire fine-tuning efficacement. Ces types de données peuvent aussi être utilisé pour rendre un modèle pré-entrainé plus petit sans QLoRA.

On peut charger un modèle transformers en 4-bit comme ça :
```
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", load_in_4bit=True, device_map="auto")
```

## Adapter fine-tuning

Ce n'est pas possible de faire un véritable entrainement sur un modèle quantifié. Cependant, il est possible de finetuné des modèles quantifiés avec des méthodes **PEFT (parameter efficient fine tuning)** et d'entrainer des adapeteurs au dessus de ces modèles. La méthode de finetuning va être basé sur une méthode appelé **LoRA (Low Rank Adapters)** : au lieu de finetuné le modèle entier, on va juste juste finetuner ces adapeteurs et les charger proprement dans le modèle.

**QLoRA** réduit la mémoire utilisé lors des finetuning de LLM sans contreparti niveau performances comparé au finetuning de modèle en 16-bit standard. Cette méthode permer un finetuning d'un modèle de 33B  sur un GPU 24GB et un finetuning d'un modèle 65B sur un GPU 46GB. 

QLoRA utilise la quantifiquation 4-bit pour compressé un LLM préentrainé. Les paramètres du LM sont ensuite gelé et un petit nombre de paramètres entrainables sont ajoutés au modèle sous la forme de **Low-Rank Adapaters**. Durant le finetuning, QLoRA rétropage les gardient au travers du modèle LM prétentrainé quantifié 4-bit gelé jusqu'au Low-Rank Adapters. Les couches LoRA sont les seuls paramètres qui sont mise à jours surant l'entrainement. [Papier QLoRA](https://arxiv.org/abs/2305.14314).

```
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
```
On charge le modèle en 4-bit
```
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", load_in_4bit=True, device_map="auto")
```

On peut jouer avec différentes varaiationd e quantifiquation 4-bits comme NF4 (normalized float 4) ou la pure quantifiquation FP4. il est recommendé d'utiliser la quantifiquation NF4 pour de meilleurs performances.

Exemple pour charger un modèle en 4bits en utilisant la quantifiquation NF4 avec une double quantiifquation avec le dtype de calcul en bfloat16 pour un entrainement plus rapide :
```
from transformers import BitsAndBytesConfig
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,  # Permet une seconde quantifiquation après la première pour sauvegarder en plus 0.4 bits par paramètre
   bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
```

On utilise la double quantifiquation quand on a des problèpmpes d emémoires, la quantifiquation NF4 pour une méilleure précision et le dtype 16-bit pour un finetuning plus rapide.

Il n'est pas possible de faire un pure entrainement 4-bit sur ces modèles. Cependant, on peut entrainer ces modèles en tirant parti des méthodes PEFT et en entrainant des adapteurs au dessus d'eux. 
