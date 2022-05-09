{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "uniavan.ipynb",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "#Importação das bibliotecas"
      ],
      "metadata": {
        "id": "964FrStZ7R6p"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 47,
      "metadata": {
        "id": "HrhRYRH1s2hU"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import spacy\n",
        "import re\n",
        "import string\n",
        "import random\n",
        "import matplotlib.pyplot as plt"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%%capture\n",
        "!python -m spacy download pt"
      ],
      "metadata": {
        "id": "fIg1ONYXurFw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Leitura e ajuste do conjunto de dados"
      ],
      "metadata": {
        "id": "rAdPBvqh7aMI"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train = pd.read_csv('/content/drive/MyDrive/textMining/Train50.csv', delimiter=';')\n",
        "\n",
        "train.head()\n",
        "train.drop(['id', 'tweet_date', 'query_used'], axis=1, inplace=True)"
      ],
      "metadata": {
        "id": "uUvPvaWRu_93"
      },
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Pré-processamento do texto"
      ],
      "metadata": {
        "id": "jURo4MJH7iN2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "nlp = spacy.load('pt')\n",
        "\n",
        "stop_words = spacy.lang.pt.stop_words.STOP_WORDS\n",
        "\n",
        "def pre_processing(texto):\n",
        "    # Letras minúsculas\n",
        "    texto = texto.lower()\n",
        "\n",
        "    # Nome do usuário\n",
        "    texto = re.sub(r'@[A-Za-z0-9$-_@.&+]+', ' ', texto)\n",
        "\n",
        "    #URLs\n",
        "    texto = re.sub(r'https?://[A-Za-z0-9./]+', ' ', texto)\n",
        "\n",
        "    #Espaços em branco\n",
        "    texto = re.sub(r' +', ' ', texto)\n",
        "\n",
        "    # Emoticons\n",
        "    lista_emocoes = {':)': 'emocaopositiva',\n",
        "                    ':d': 'emocaopositiva',\n",
        "                    ':(': 'emocaonegativa'}\n",
        "    for emocao in lista_emocoes:\n",
        "        texto = texto.replace(emocao, lista_emocoes[emocao])\n",
        "\n",
        "    # Lematização\n",
        "    documento = nlp(texto)\n",
        "\n",
        "    lista = []\n",
        "    for token in documento:\n",
        "        lista.append(token.lemma_)\n",
        "\n",
        "    # Stop words\n",
        "    lista = [palavra for palavra in lista if palavra not in stop_words]\n",
        "\n",
        "    # Pontuações\n",
        "    lista = [palavra for palavra in lista if palavra not in string.punctuation]\n",
        "\n",
        "    # Remove números e concatena a lista\n",
        "    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])\n",
        "\n",
        "    return lista"
      ],
      "metadata": {
        "id": "I5uD5SB1vfQA"
      },
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train['tweet_text'] = train['tweet_text'].apply(pre_processing)"
      ],
      "metadata": {
        "id": "-7254Scevyvv"
      },
      "execution_count": 33,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Ajuste das categorias"
      ],
      "metadata": {
        "id": "_prSWiPn7n5t"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_final = []\n",
        "for texto, emocao in zip(train['tweet_text'], train['sentiment']):\n",
        "    if emocao == 1:\n",
        "        dic = ({'POSITIVO': True, 'NEGATIVO': False})\n",
        "    elif emocao == 0:\n",
        "        dic = ({'POSITIVO': False, 'NEGATIVO': True})\n",
        "\n",
        "    train_final.append([texto, dic.copy()])"
      ],
      "metadata": {
        "id": "ie1EhfmQyVSq"
      },
      "execution_count": 34,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Treinamento do modelo"
      ],
      "metadata": {
        "id": "HNllPMHF7urV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model = spacy.blank('pt')\n",
        "categories = model.create_pipe('textcat')\n",
        "categories.add_label('POSITIVO')\n",
        "categories.add_label('NEGATIVO')\n",
        "model.add_pipe(categories)"
      ],
      "metadata": {
        "id": "qYepaDV_yisl"
      },
      "execution_count": 38,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "historico = []\n",
        "model.begin_training()\n",
        "for epoc in range(20):\n",
        "    random.shuffle(train_final)\n",
        "    losses = {}\n",
        "    for batch in spacy.util.minibatch(train_final, 512):\n",
        "        texts = [model(text) for text, entities in batch]\n",
        "        annotations = [{'cats': entities} for texto, entities in batch]\n",
        "        model.update(texts, annotations, losses=losses)\n",
        "        historico.append(losses)\n",
        "    if epoc % 5 == 0:\n",
        "        print(losses)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HwdBJw3TzGco",
        "outputId": "c696c4a6-1978-425e-89cd-f57159e707a3"
      },
      "execution_count": 41,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{'textcat': 6.679298815107093e-06}\n",
            "{'textcat': 2.555433579209433e-07}\n",
            "{'textcat': 6.370858549468506e-07}\n",
            "{'textcat': 3.176723186277828e-07}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.to_disk('/content/drive/MyDrive/textMining/spc_model')"
      ],
      "metadata": {
        "id": "_rxGdXtl8lqt"
      },
      "execution_count": 49,
      "outputs": []
    }
  ]
}