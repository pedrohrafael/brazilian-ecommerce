# Análise de sentimentos 
Analise de Sentimentos para reviews é um projeto baseado em Python que visa extrair insights valiosos de avaliações de clientes em uma plataforma de ecommerce. Usando técnicas de **Processamento de Linguagem Natural (NLP)**, este projeto analisa o sentimento expresso nas avaliações, ajudando as empresas a entender o feedback do cliente e tomar decisões baseadas em dados para melhorar seus produtos e serviços.

## Principais benefícios:

- **Compreensão do sentimento do cliente:** a análise de sentimento permite que as empresas entendam o sentimento geral dos clientes em relação a seus produtos ou serviços. Ao classificar automaticamente as - avaliações como positivas, negativas ou neutras, as empresas podem avaliar a satisfação do cliente e identificar as áreas que precisam ser melhoradas.

- **Melhoria do produto:** ao analisar os sentimentos nas avaliações, as empresas podem identificar pontos problemáticos comuns ou áreas de elogios para seus produtos. Essas informações podem ser inestimáveis para as equipes de desenvolvimento de produtos fazerem melhorias e otimizarem recursos que se alinhem com as preferências do cliente.

- **Melhorias no atendimento ao cliente:** identificar sentimentos negativos nas avaliações pode destacar problemas de atendimento ao cliente. Ao abordar prontamente as reclamações dos clientes, as empresas podem melhorar sua reputação e promover a fidelidade do cliente.

- **Tomada de decisão baseada em dados:** a análise de sentimento fornece dados quantificáveis para dar suporte aos processos de tomada de decisão. Essa abordagem orientada por dados ajuda as empresas a priorizar tarefas e alocar recursos com eficiência com base no feedback do cliente.

- **Retenção e aquisição de clientes:** entender os sentimentos dos clientes pode orientar as estratégias de retenção de clientes. Clientes satisfeitos têm maior probabilidade de permanecer leais, enquanto lidar com sentimentos negativos pode evitar a rotatividade. Além disso, avaliações positivas podem atrair novos clientes e impulsionar a aquisição.

## Aspectos técnicos
- **Pré-processamento de texto:** os dados de texto extraídos serão pré-processados para remover o ruído, incluindo URLs, acentos, dígitos, caracteres especiais e stopwords, além da aplicação de Lemmatização.

- **Análise de sentimento:** o núcleo do projeto envolve o emprego de técnicas de NLP e modelos de aprendizado de máquina pré-treinados (por exemplo, Naive Bayes, Support Vector Machines ou modelos baseados em aprendizado profundo como BERT) para classificar o sentimento de cada revisão.

- **Interface do usuário:** para uma versão avançada do projeto, uma interface de usuário simples baseada na web foi desenvolvida usando [gradio.app](https://www.gradio.app/), permitindo que os usuários insiram texto personalizado e recebam resultados de análise de sentimento em tempo real.

### Requirements
#

```
pip install -r requirements.txt
```

### Organização do projeto
#
```
2. Análise de sentimentos
├─ models
├─ src
│  ├─ dataset
│  ├─ models
│  └─ preprocess
├─ app.py
├─ README.md
└─ requirements.txt
```

## Gradio App
```
python app.py
```

<p align="center">
  <img src="https://github.com/pedrohrafael/brazilian-ecommerce/assets/59976208/e4e27444-97a2-49d8-8bfa-d9febd3926cc" style="width:75% ;align:center"/>
<p/>

