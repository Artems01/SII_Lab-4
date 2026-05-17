import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. ЗАГРУЗКА ДАННЫХ
train = pd.read_csv('tests/train.csv')
test = pd.read_csv('tests/test.csv')
sub = pd.read_csv('tests/submission.csv')

features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']
X_train = train[features]
y_train = train['Y']
X_test = test[features]
y_test = sub['Y']

print('Классы в обучающей выборке:')
print(y_train.value_counts())
print('\nКлассы в тестовой выборке:')
print(y_test.value_counts())

# 2. СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛЕЙ
model_nusvc = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    NuSVC(nu=0.1, kernel='rbf', random_state=42)
)
model_nusvc.fit(X_train, y_train)

model_percept = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    Perceptron(random_state=42)
)
model_percept.fit(X_train, y_train)

# 3. ОЦЕНКА НА TRAIN И TEST (NuSVC)
print('\n' + '='*30 + '\nРЕЗУЛЬТАТЫ ДЛЯ NuSVC\n' + '='*30)
pred_train_nusvc = model_nusvc.predict(X_train)
train_acc_nusvc = accuracy_score(y_train, pred_train_nusvc)
print('Отчет об обучающей выборке (NuSVC):')
print(classification_report(y_train, pred_train_nusvc, zero_division=0))
cm_train_nusvc = confusion_matrix(y_train, pred_train_nusvc)

pred_test_nusvc = model_nusvc.predict(X_test)
test_acc_nusvc = accuracy_score(y_test, pred_test_nusvc)
print('Отчет о тестовой выборке (NuSVC):')
print(classification_report(y_test, pred_test_nusvc, zero_division=0))
cm_test_nusvc = confusion_matrix(y_test, pred_test_nusvc)

# 4. ОЦЕНКА НА TRAIN И TEST (Perceptron)
print('\n' + '='*30 + '\nРЕЗУЛЬТАТЫ ДЛЯ PERCEPTRON\n' + '='*30)
pred_train_percept = model_percept.predict(X_train)
train_acc_percept = accuracy_score(y_train, pred_train_percept)
print('Отчет об обучающей выборке (Perceptron):')
print(classification_report(y_train, pred_train_percept, zero_division=0))
cm_train_percept = confusion_matrix(y_train, pred_train_percept)

pred_test_percept = model_percept.predict(X_test)
test_acc_percept = accuracy_score(y_test, pred_test_percept)
print('Отчет о тестовой выборке (Perceptron):')
print(classification_report(y_test, pred_test_percept, zero_division=0))
cm_test_percept = confusion_matrix(y_test, pred_test_percept)

# 5. ВАЖНОСТЬ ПРИЗНАКОВ (Для обеих моделей)
imp_nusvc = permutation_importance(model_nusvc, X_train, y_train, n_repeats=10, random_state=42, scoring='accuracy')
feat_imp_nusvc = imp_nusvc.importances_mean

imp_percept = permutation_importance(model_percept, X_train, y_train, n_repeats=10, random_state=42, scoring='accuracy')
feat_imp_percept = imp_percept.importances_mean

# 6. ГРАФИКИ (РОВНО 7 ОКОН ПО ОЧЕРЕДИ)
# Окно 1: Матрица ошибок NuSVC (Обучающая выборка)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_train_nusvc, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Предсказано')
plt.ylabel('Настоящее')
plt.title('Матрица ошибок NuSVC (Обучающая выборка)')
plt.tight_layout()
plt.show()

# Окно 2: Матрица ошибок NuSVC (Тестовая выборка)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_test_nusvc, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Предсказано')
plt.ylabel('Настоящее')
plt.title('Матрица ошибок NuSVC (Тестовая выборка)')
plt.tight_layout()
plt.show()

# Окно 3: Матрица ошибок Perceptron (Обучающая выборка)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_train_percept, annot=True, fmt='d', cmap='Oranges', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Предсказано')
plt.ylabel('Настоящее')
plt.title('Матрица ошибок Perceptron (Обучающая выборка)')
plt.tight_layout()
plt.show()

# Окно 4: Матрица ошибок Perceptron (Тестовая выборка)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_test_percept, annot=True, fmt='d', cmap='Oranges', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Предсказано')
plt.ylabel('Настоящее')
plt.title('Матрица ошибок Perceptron (Тестовая выборка)')
plt.tight_layout()
plt.show()

# Окно 5: Важность признаков NuSVC
plt.figure(figsize=(8, 5))
plt.bar(X_train.columns, feat_imp_nusvc, color='steelblue')
plt.xlabel('Признаки')
plt.ylabel('Важность')
plt.title('Важность признаков (NuSVC)')
plt.tight_layout()
plt.show()

# Окно 6: Важность признаков Perceptron
plt.figure(figsize=(8, 5))
plt.bar(X_train.columns, feat_imp_percept, color='darkorange')
plt.xlabel('Признаки')
plt.ylabel('Важность')
plt.title('Важность признаков (Perceptron)')
plt.tight_layout()
plt.show()

# Окно 7: Сравнение точности моделей
plt.figure(figsize=(8, 5))
vals = [train_acc_nusvc, test_acc_nusvc, train_acc_percept, test_acc_percept]
labels = ['NuSVC Train', 'NuSVC Test', 'Percept Train', 'Percept Test']
bars = plt.bar(labels, vals, color=['steelblue', 'lightblue', 'darkorange', 'moccasin'])
plt.ylim(0, 1.1)
plt.ylabel('Точность')
plt.title('Точность моделей на датасетах')
for bar, val in zip(bars, vals):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f'{val:.3f}', ha='center')
plt.tight_layout()
plt.show()