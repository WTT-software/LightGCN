# Opis projektu – rekomendacje z użyciem LightGCN i NGCF

Celem projektu jest zbudowanie modelu, który potrafi polecać użytkownikom produkty (tu: książki z Amazon-book) na podstawie ich wcześniejszych interakcji. Zakładamy dane typu **implicit feedback** – wiemy tylko, że użytkownik coś obejrzał / kliknął / kupił, ale nie mamy ocen w skali (np. 1–5).

---

## Problem

Chcemy odpowiedzieć na pytanie:

> Które książki warto pokazać użytkownikowi, aby było duże prawdopodobieństwo, że go zainteresują?

Dane przedstawiamy jako **graf dwudzielny**:
- jeden typ węzłów to użytkownicy,
- drugi typ to książki (itemy),
- krawędź między użytkownikiem a książką oznacza interakcję (np. zakup).

---

## Dotychczasowe rozwiązanie: NGCF

**NGCF (Neural Graph Collaborative Filtering)** to model, który:

- wykorzystuje strukturę grafu użytkownik–item,
- w każdej warstwie:
  - zbiera informację od sąsiadów w grafie,
  - przekształca embeddingi macierzami wag,
  - stosuje nieliniowości (np. ReLU).

Zalety:
- dobrze wykorzystuje informacje o relacjach w grafie,
- potrafi modelować złożone wzorce współwystępowania użytkowników i itemów.

Wady:
- dużo parametrów (wiele macierzy wag),
- większa złożoność obliczeniowa,
- większe ryzyko przeuczenia,
- trudniejsze strojenie i interpretacja.

---

## Nowe podejście: LightGCN

**LightGCN** opiera się na obserwacji, że w zadaniu rekomendacji na grafie najważniejsza jest sama **propagacja po grafie**, a nie rozbudowane „neuronowe” bloki.

W praktyce oznacza to:

- jeden wspólny zbiór embeddingów dla wszystkich węzłów (użytkownicy + itemy),
- kolejne kroki propagacji to po prostu mnożenie przez znormalizowaną macierz sąsiedztwa,
- brak:
  - dodatkowych macierzy wag w warstwach,
  - funkcji aktywacji,
  - złożonych operacji łączenia.

Końcowy embedding użytkownika (lub itemu) jest średnią embeddingów z kilku kroków propagacji (kilku „warstw”).

Model uczony jest z użyciem funkcji kosztu **BPR (Bayesian Personalized Ranking)**:
- dla każdego użytkownika bierzemy:
  - item pozytywny (z interakcji),
  - item negatywny (losowy, bez interakcji),
- model ma nadać wyższy wynik itemowi pozytywnemu niż negatywnemu.

---

## Co jest zrobione w projekcie

W projekcie:

1. Wykorzystano podzbiór zbioru **Amazon-book**:
   - filtrowanie użytkowników z małą liczbą interakcji,
   - ograniczenie liczby użytkowników,
   - ponowne nadanie ID (ciągłe indeksy od 0).

2. Dane podzielono na:
   - zbiór treningowy,
   - zbiór walidacyjny,
   - zbiór testowy.

3. Zbudowano graf użytkownik–item:
   - utworzono znormalizowaną macierz sąsiedztwa,
   - przygotowano struktury do uczenia na grafie.

4. Zaimplementowano model **LightGCN** w TensorFlow 2:
   - propagacja embeddingów po grafie,
   - uczenie z BPR i negatywnym samplingiem.

5. Przeprowadzono ewaluację:
   - metryki **Recall@K** i **NDCG@K** (np. K = 20),
   - ocena jakości rekomendacji na zbiorze walidacyjnym i testowym.

Projekt pokazuje, jak uproszczenie modelu z NGCF do LightGCN wpływa na:
- liczbę parametrów,
- szybkość obliczeń,
- jakość rekomendacji w praktycznym scenariuszu z danymi implicit.
