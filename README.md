# Processo Seletivo – Intensivo Maker | AI

### 👤 Identificação
**Nome:** Gabriel Souza Santos

---

## 1️⃣ Resumo da Arquitetura do Modelo

O modelo foi implementado utilizando uma Rede Neural Convolucional (CNN) leve, projetada para a classificação de dígitos
do dataset MNIST.

A arquitetura é composta por:

- 3 camadas Conv2D com filtros progressivos (16 → 24 → 32), utilizando ativação ReLU e kernels 3x3
- 2 camadas MaxPooling2D para redução da dimensionalidade espacial
- Camada Flatten para transformação dos mapas de características em vetor 1D
- Camada densa intermediária com 32 neurônios e ativação ReLU
- Camada de saída com 10 neurônios e ativação softmax para classificação multiclasse

A arquitetura foi intencionalmente mantida compacta, sem uso de Batch Normalization ou Dropout, visando adequação a cenários de Edge AI com restrições computacionais e execução eficiente em CPU.

---

## 2️⃣ Bibliotecas Utilizadas

TensorFlow >= 2.12 NumPy O TensorFlow foi utilizado para construção, treinamento e conversão do modelo, enquanto o NumPy
foi mantido para suporte a operações numéricas básicas e compatibilidade com o pipeline do projeto.

---

## 3️⃣ Técnica de Otimização do Modelo

A otimização foi realizada utilizando o TensorFlow Lite Converter com aplicação da técnica de:

Dynamic Range Quantization (`tf.lite.Optimize.DEFAULT`)

Essa técnica reduz o tamanho do modelo ao quantizar os pesos durante a conversão para TensorFlow Lite, diminuindo o
consumo de memória e melhorando a eficiência computacional durante inferência.

---

## 4️⃣ Resultados Obtidos

- Acurácia no conjunto de teste: ~98.5%
- Loss no conjunto de teste: ~3.5%
- Tamanho do modelo Keras (.h5): ~280 KB
- Tamanho do modelo otimizado (.tflite): ~28 KB

A conversão para TensorFlow Lite resultou em uma redução significativa de tamanho (~10x), mantendo desempenho adequado
para o problema de classificação.

---

## 5️⃣ Comentários Adicionais

Durante o desenvolvimento, foram realizadas decisões técnicas com foco em eficiência e restrições de ambiente:

- A arquitetura foi mantida simples para evitar overfitting e garantir compatibilidade com Edge AI
- Testes com Dropout foram avaliados, mas não apresentaram melhoria significativa neste cenário 
- O modelo foi treinado com 4 épocas, respeitando o limite de execução eficiente em ambiente automatizado 
- Um dos principais desafios foi o balanceamento entre tamanho do modelo e acurácia, priorizando eficiência
computacional sem perda relevante de desempenho
- A camada densa originalmente maior foi reduzida para 32 neurônios, resultando em um modelo mais leve sem impacto
significativo na performance