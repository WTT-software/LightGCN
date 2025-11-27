# LightGCN – implementacja w TensorFlow 2 na zbiorze Amazon-book

Notebook: `LightGCN.ipynb`  
Środowisko docelowe: Google Colab (GPU opcjonalne, ale zalecane).

Implementacja modelu **LightGCN** do rekomendacji na danych implicit-feedback (interakcje użytkownik–item, bez ratingów). Model jest trenowany na podzbiorze zbioru **Amazon-book** z oryginalnego repozytorium LightGCN.

---

## 1. Funkcjonalność

Notebook realizuje:

1. Pobranie i wczytanie danych `Amazon-book` z repozytorium LightGCN.
2. Zbudowanie zredukowanego podzbioru użytkowników i itemów:
   - filtr po minimalnej liczbie interakcji na użytkownika,
   - losowy wybór maksymalnej liczby użytkowników,
   - reindeksacja ID użytkowników i itemów do zakresu ciągłego.
3. Podział interakcji na:
   - zbiór treningowy,
   - zbiór walidacyjny,
   - zbiór testowy.
4. Zbudowanie znormalizowanej macierzy sąsiedztwa grafu user–item.
5. Zdefiniowanie i trenowanie modelu **LightGCN** w TensorFlow 2:
   - pojedynczy tensor embeddingów dla wszystkich węzłów (user + item),
   - propagacja z użyciem znormalizowanej macierzy sąsiedztwa,
   - trenowanie z funkcją kosztu BPR.
6. Ewaluację z użyciem metryk:
   - `Recall@K` (domyślnie K=20),
   - `NDCG@K`.
7. Generowanie rekomendacji top-K dla wybranego użytkownika.

---

## 2. Wymagania

Notebook zakłada środowisko z:

- Python 3.x
- TensorFlow 2.x (`tf.keras`, `tf.sparse`)
- NumPy
- Matplotlib

W Google Colab nic poza TensorFlow nie wymaga dodatkowej instalacji.  
GPU jest opcjonalne, ale przyspiesza propagację grafową i trening.

---

## 3. Dane: Amazon-book

Notebook pobiera dane z repozytorium:

- Źródło: `https://github.com/kuandeng/LightGCN`
- Używany podkatalog: `Data/amazon-book`

Pobierane pliki:

- `train.txt` – interakcje treningowe,
- `test.txt` – interakcje do ewaluacji,
- `user_list.txt`, `item_list.txt` – mapowanie oryginalnych ID na indeksy liczbowe.

Ścieżka docelowa w Colabie:

- `/content/data/amazon-book`

Wczytywanie danych realizuje funkcja:

- `load_amazon_book_train(...)`

W wyniku otrzymywane są m.in.:

- liczba użytkowników `num_users`,
- liczba itemów `num_items`,
- wszystkie interakcje w postaci par `(user_id, item_id)`,
- słownik `user_pos_items_all` z pozytywnymi itemami dla każdego użytkownika.

---

## 4. Redukcja zbioru i podział na zbiory

### 4.1. Zmniejszenie zbioru

Funkcja:

- `build_subset(user_pos_items_all, max_users, min_interactions)`

realizuje:

- odrzucenie użytkowników z liczbą interakcji < `min_interactions`,
- losowy wybór maksymalnie `max_users` użytkowników,
- reindeksację użytkowników i itemów do zakresu `0..num_users_sub-1` oraz `0..num_items_sub-1`,
- zbudowanie przekrojowego słownika interakcji dla podzbioru.

Zwracane wartości:

- `num_users_sub`, `num_items_sub`,
- listy z interakcjami (`user_ids_sub`, `item_ids_sub`),
- słownik `user_pos_items_sub`.

Notebook wywołuje tę funkcję, żeby pracować na mniejszym, gęstszym podzbiorze Amazon-book, co przyspiesza eksperymenty.

### 4.2. Podział na train / val / test

Funkcja:

- `split_user_interactions(user_pos_items)`

dzieli interakcje dla każdego użytkownika na trzy zbiory:

- jeśli użytkownik ma mniej niż 3 interakcje:
  - wszystkie trafiają do train,
- w przeciwnym razie:
  - 1 item losowo → zbiór test,
  - 1 item losowo → zbiór walidacyjny,
  - pozostałe → zbiór treningowy.

Zwracane słowniki:

- `train_user_pos_items`,
- `val_user_pos_items`,
- `test_user_pos_items`.

Do pomocy w budowie list par `(u, i)` dla grafu używana jest funkcja:

- `dict_to_arrays(user_pos_dict)`

---

## 5. Macierz sąsiedztwa grafu user–item

Znormalizowaną macierz sąsiedztwa grafu dwudzielnego user–item buduje funkcja:

- `build_normalized_adj(num_users, num_items, user_ids, item_ids)`

Cechy:

- graf dwudzielny, węzły:
  - użytkownicy: `0 .. num_users-1`,
  - itemy: `num_users .. num_users + num_items - 1`,
- dla każdej pary `(u, i)` z datosetu treningowego tworzona jest krawędź w dwie strony,
- macierz jest symetrycznie normalizowana (`D^{-1/2} A D^{-1/2}`),
- wynik to `tf.sparse.SparseTensor` o rozmiarze `(num_users + num_items, num_users + num_items)`.

Macierz jest budowana wyłącznie z danych treningowych.

---

## 6. Model: klasa `LightGCN`

Model jest zdefiniowany jako:

- klasa `LightGCN(tf.keras.Model)`

Kluczowe elementy:

- parametry wejściowe:
  - `num_users`,
  - `num_items`,
  - `embedding_dim`,
  - `num_layers`,
  - znormalizowana macierz sąsiedztwa `adj`,
- pojedynczy trenowalny tensor:
  - `node_embeddings` o wymiarze `(num_users + num_items, embedding_dim)`,
- brak dodatkowych wag liniowych w warstwach – zgodnie z ideą LightGCN.

Propagacja informacji w grafie odbywa się w metodzie:

- `propagate(self)`

Logika propagacji:

- iteracyjne rozmnażanie embeddingów po grafie przez kolejne mnożenia `adj * E`,
- sumowanie embeddingów z kolejnych „warstw”,
- uśrednianie po wszystkich krokach (`(E^{(0)} + ... + E^{(K)}) / (K+1)`),
- rozdzielenie końcowych embeddingów na:
  - `user_embs` (pierwsze `num_users` wierszy),
  - `item_embs` (pozostałe wiersze).

Notebook tworzy instancję modelu z typowymi parametrami, np.:

- `embedding_dim = 64`,
- `num_layers = 3`.

Optimizer:

- `tf.keras.optimizers.Adam` z zadaną wartością `learning_rate` (np. `1e-3`).

---

## 7. Funkcja kosztu: BPR + regularyzacja

Trenowanie odbywa się z użyciem funkcji kosztu:

- `bpr_loss(model, user_ids, pos_item_ids, neg_item_ids, l2_reg)`

Główne elementy:

- wykorzystanie embeddingów po propagacji (`propagate`),
- obliczanie skalarnych produktów:
  - `score(u, i_pos)` oraz `score(u, i_neg)`,
- BPR:
  - minimalizacja `-log σ(score_pos - score_neg)`,
- regularyzacja L2:
  - liczona na tzw. „ego-embeddingach” (`node_embeddings`) dla aktualnie użytych użytkowników i itemów.

---

## 8. Negatywny sampling

Batchowanie dla BPR realizuje funkcja:

- `make_bpr_sampler(user_pos_items, num_items, batch_size)`

Zwraca ona wewnętrzną funkcję:

- `sample_batch()`

która generuje batch:

- `user_ids`,
- `pos_item_ids`,
- `neg_item_ids`,

zgodnie z zasadą:

- pozytywny item dla użytkownika jest wybierany z jego obserwowanych interakcji,
- negatywny item jest losowany z całej puli itemów z wykluczeniem pozytywów tego użytkownika.

---

## 9. Trening: `train_step` i pętla epok

Krok optymalizacji jest opakowany w funkcję:

- `train_step(model, user_ids, pos_item_ids, neg_item_ids)`

W skrócie:

- używa `tf.GradientTape`,
- liczy `bpr_loss`,
- oblicza gradienty względem trenowalnych zmiennych modelu,
- stosuje `optimizer.apply_gradients(...)`.

Pętla treningowa w notebooku:

- liczba epok: `num_epochs` (np. 1000),
- kroki na epokę: `steps_per_epoch` (np. 1000),
- w każdej epoce:
  - wielokrotne wywołanie `sample_batch()` oraz `train_step(...)`,
  - obliczanie średniego lossu dla epoki.

Co kilka epok (np. co `validate_every`) wywoływana jest ewaluacja na zbiorze walidacyjnym.

---

## 10. Ewaluacja: `evaluate_topk`

Metryki jakości rekomendacji liczone są w funkcji:

- `evaluate_topk(model, user_pos_train, user_pos_eval, num_items, K)`

Metoda:

1. Obliczenie embeddingów użytkowników i itemów przez `propagate`.
2. Dla każdego użytkownika z `user_pos_eval`:
   - obliczenie scoringu dla wszystkich itemów,
   - wykluczenie itemów ze zbioru treningowego (`user_pos_train`),
   - wybór top-K itemów,
   - policzenie:
     - `Recall@K` (pokrycie pozytywów w top-K),
     - `NDCG@K` (ważenie pozycji trafień).
3. Uśrednienie metryk po użytkownikach.

Wykorzystywana jest:

- do walidacji podczas treningu (np. `Recall@20`, `NDCG@20`),
- do końcowej ewaluacji na zbiorze testowym (przy podmianie `user_pos_eval` na słownik testowy).

---

## 11. Rekomendacje dla pojedynczego użytkownika

Notebook zawiera wygodną funkcję do generowania rekomendacji top-K:

- `recommend_for_user(model, u, train_user_pos_items, extra_exclude_items, topk)`

Zwraca ona listę indeksów itemów rekomendowanych użytkownikowi `u`:

- na podstawie końcowych embeddingów,
- z wykluczeniem itemów, które użytkownik już posiada w treningu,
- opcjonalnie z dodatkową listą itemów do wykluczenia (`extra_exclude_items`).

---

## 12. Sposób użycia (Google Colab)

1. Załaduj notebook `LightGCN.ipynb` do Google Colab.
2. Włącz GPU (Menu: `Runtime` → `Change runtime type` → `GPU`).
3. Uruchom komórki po kolei:
   - importy i pobranie danych,
   - wczytanie i redukcja zbioru (`load_amazon_book_train`, `build_subset`),
   - podział na train/val/test (`split_user_interactions`),
   - budowa macierzy sąsiedztwa (`build_normalized_adj`),
   - definicja modelu (`LightGCN`),
   - trening (`make_bpr_sampler`, `train_step` + pętla epok),
   - ewaluacja (`evaluate_topk`),
   - przykładowe rekomendacje (`recommend_for_user`).

---

## 13. Odniesienia

- Paper:  
  X. He et al., **“LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation”**, SIGIR 2020.
- Oryginalne repozytorium:  
  `https://github.com/kuandeng/LightGCN`
- Zbiór danych Amazon-book:  
  katalog `Data/amazon-book` w repozytorium powyżej.
