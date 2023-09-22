#!/usr/bin/env python
# coding: utf-8

# # Módulo 07 - Tarefa 02
# 
# #### 1) Carregue a base e garanta que a base está como deveria.
# 
# Considere a base que você ajustou na lição passada. Carregue-a. Caso ainda haja alguma pendência, ajuste - lembre-se de que o scikitlearn não recebe variáveis em formato string, somente numéricas, e não aceita '*missings*'. 
# 
# - Separe 70% da base para treinamento e 30% para validação. Cada uma dessas partes terá dois objetos, um armazenando a variável resposta ```mau```e outro armazenando as variáveis explicativas (lembrando: sem variáveis string, já com as dummies).

# In[1]:


import pandas as pd

# Carregue a base de dados demo01.csv
data = pd.read_csv('demo01.csv')

# Verifique se há valores faltantes
missing_values = data.isnull().sum()
print("Valores faltantes por coluna:")
print(missing_values)

# Se houver valores faltantes, você pode escolher como tratá-los, por exemplo, preenchendo com a média ou removendo as linhas.
# Vamos preencher com a média neste exemplo.
data = data.fillna(data.mean())

# Verifique novamente se há valores faltantes após o preenchimento
missing_values = data.isnull().sum()
print("\nValores faltantes após o preenchimento com a média:")
print(missing_values)

# Agora, vamos converter as variáveis categóricas em variáveis dummy
categorical_columns = data.select_dtypes(include=['object']).columns

# Crie variáveis dummy para as colunas categóricas
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Verifique o DataFrame atualizado
print("\nDataFrame atualizado:")
print(data.head())

# Agora, seus dados devem estar pré-processados e prontos para serem usados com o scikit-learn.


# In[2]:


# Verifique as colunas do DataFrame atualizado
print("Colunas presentes no DataFrame:")
print(data.columns)

# Verifique se a coluna 'target' está presente
if 'target' in data.columns:
    print("\nA coluna 'target' está presente no DataFrame.")
else:
    print("\nA coluna 'target' não está presente no DataFrame.")



# In[3]:


from sklearn.model_selection import train_test_split

# Defina as variáveis explicativas (features) e a variável de resposta (target)
X = data.drop(columns=['ID', 'MONTHS_BALANCE'])  # Variáveis explicativas (excluindo colunas não desejadas)
y = data['STATUS_1']  # Variável de resposta (substitua 'STATUS_1' pelo nome da coluna desejada)

# Divida os dados em conjuntos de treinamento (70%) e validação (30%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Agora você tem quatro conjuntos de dados:
# X_train - Variáveis explicativas de treinamento
# y_train - Variável de resposta de treinamento
# X_val   - Variáveis explicativas de validação
# y_val   - Variável de resposta de validação


# #### 2) Vamos para o modelo:
# 
# 1. Defina um objeto com a função da árvore de decisão vista em aula.
# 2. Treine o modelo com os dados que você separou para treinamento.
# 3. Visualize a árvore. Talvez você queira aumentar um pouco a figura.
# 4. Produza uma visualização da matriz de classificação (ou matriz de confusão) - coloque os rótulos como "aprovados" e "reprovados" - pois afinal de contas, são essas as decisões que se podem tomar com propostas de crédito.
# 5. Calcule a acurácia na base de treinamento

# In[4]:


from sklearn.tree import DecisionTreeClassifier

# Crie um objeto de Árvore de Decisão
tree_classifier = DecisionTreeClassifier(random_state=42)

# Agora você tem um objeto `tree_classifier` que representa um modelo de Árvore de Decisão.


# In[5]:


# Treine o modelo com os dados de treinamento
tree_classifier.fit(X_train, y_train)

# Agora, o modelo de Árvore de Decisão está treinado com os dados de treinamento.


# In[7]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Calcular a matriz de confusão
y_pred = tree_classifier.predict(X_val)
cm = confusion_matrix(y_val, y_pred)

# Definir os rótulos personalizados
class_names = ["aprovados", "reprovados"]

# Plotar a matriz de confusão usando Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Previsões')
plt.ylabel('Valores Verdadeiros')
plt.title("Matriz de Confusão")
plt.show()


# In[8]:


# Calcular a acurácia nos dados de treinamento
train_accuracy = tree_classifier.score(X_train, y_train)

print("Acurácia nos dados de treinamento:", train_accuracy)


# #### 3) Vamos avaliar o modelo na base de testes
# 
# 1. Classifique a base de teste de acordo com a árvore que você treinou no item 2.
# 2. Produza a visualização da matriz de confusão para a base de teste.
# 3. Calcule a acurácia da base de teste. Compare com a acurácia da base de treinamento.
# 4. Treine uma nova árvore com número mínimo de observações por folha de 5 e máximo de profundidade de 10. Use o random_state = 123. Avalie a matriz de classificação. Observe a distribuição da predição - qual a proporção de proponentes foram classificados como 'maus'?
# 5. Como ficaria a acurácia se você classificasse todos os contratos como 'bons'?

# In[11]:


# Divida os dados em conjuntos de treinamento (70%) e validação (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[13]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Suponha que você já tenha seus dados de treinamento e teste em X_train, X_test, y_train e y_test

# Crie um objeto de Árvore de Decisão
tree_classifier = DecisionTreeClassifier(random_state=42)

# Treine o modelo com os dados de treinamento
tree_classifier.fit(X_train, y_train)

# Faça previsões na base de teste
y_pred_test = tree_classifier.predict(X_test)

# Calcular a matriz de confusão
confusion = confusion_matrix(y_test, y_pred_test)

# Configurar um gráfico de calor (heatmap) para a matriz de confusão
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Aprovados", "Reprovados"], yticklabels=["Aprovados", "Reprovados"])
plt.xlabel('Previsões')
plt.ylabel('Valores Verdadeiros')
plt.title('Matriz de Confusão')
plt.show()



# In[15]:


# Calcular a acurácia se todos os contratos forem classificados como 'bons'
acuracia_todos_bons = (y_test == 0).sum() / len(y_test)
print(f'A acurácia ao classificar todos os contratos como "bons" é: {acuracia_todos_bons:.2%}')

