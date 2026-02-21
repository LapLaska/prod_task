# ТЗ: ML-модель Look-a-Like (Retrieval → Rank) для сервиса офферов

Цель
Сервис по запросу (merchant_id, offer_id, top_n) возвращает top_n пользователей (user_id) — look-a-like аудиторию под конкретный оффер, отсортированную по score (0..1), и reasons (3 причины, feature+impact), проходящие sanity-check.

Данные
Сервис получает версии данных через API /data/batch и фиксирует их через /data/commit. Каждая версия содержит таблицы:

people (prod_clients): user_id, age_bucket, gender_cd, region, last_activity_day
segments (prizm_segments): user_id, segment, region_size, auto, traveler, entrepreneur, vip_status
transaction (prod_financial_transaction): transaction_id, user_id, merchant_id_tx, event_date, amount_bucket, online_transaction_flg, brand_dk
offer (t_offer): offer_id, merchant_id_offer, start_date, end_date, offer_text
merchant (t_merchant): merchant_id_offer, merchant_status, brand_dk
financial_account: user_id, product_cd, open_month, close_month, account_status_cd
offer_seens: user_id, offer_id, start_date, end_date
offer_activation: user_id, offer_id, activation_date
offer_reward: user_id, offer_id, event_date, reward_amt
receipts: user_id, date_operated, category_name, items_count, items_cost

Ключевые связи
1) offer.merchant_id_offer -> merchant.merchant_id_offer -> merchant.brand_dk
2) transaction.brand_dk — это “бренд” мерчанта в транзакциях
3) оффер привязывается к транзакциям через brand_dk(offer) и окно дат оффера

Обработка дат
1) Все даты парсятся в pandas.Timestamp (naive), единый формат YYYY-MM-DD.
2) end_date с пропуском или равным 5999-01-01 трактуется как “практически бесконечная” дата окончания. Во всех вычислениях end_date используется как Timestamp('5999-01-01') без дополнительных правил.
3) Границы окон включительные: [start_date, end_date] означает event_date >= start_date и event_date <= end_date.

Определения
brand_dk(offer) = merchant.brand_dk для merchant_id_offer оффера.
current_client(user, offer) = существует транзакция transaction для user_id пользователя с brand_dk == brand_dk(offer) и event_date < offer.start_date.

Конфликтность бренда
Для каждого brand_dk считаем n_merchants_for_brand = число уникальных merchant_id_offer в merchant с этим brand_dk.
brand_conflict_flg = 1 если n_merchants_for_brand > 1, иначе 0.
Никаких специальных исключений/фильтров для конфликтных брендов не делаем; эти две фичи добавляем в ранкер.

Итоговая схема модели
1) Implicit ALS на user×brand_dk (без receipts) → user_factors_norm, brand_factors_norm.
2) Эмбеддинги offer_text через deepvk/USER-base + FAISS индекс похожих офферов.
3) Генерация кандидатов (до 10k) = union(brand-sim, seed-centroid-from-similar-offers) + добивка популярностью до 10k; далее обязательный фильтр current_client.
4) Ранжирование кандидатов CatBoostRanker (YetiRank) на парных фичах (user features + retrieval features + offer features).
5) Калибровка score в (0..1) через сигмоиду score = sigmoid(a * raw + b), где a,b подбираются на валидации.
6) reasons = топ-3 фич по mean(|SHAP|) на top_n объектах ответа; impact = mean(SHAP) со знаком.

Параметры (фиксированные дефолты, задаются в params.yaml)
Общее:
random_seed = 42

Текст (deepvk/USER-base):
offer_text_max_length = 256
similar_offers_k = 50

ALS:
als_mode = "binary" или "weighted" (строго одно из двух значений)
als_factors = 128
als_iterations = 50
als_regularization = 0.01
als_alpha = 40.0
als_time_decay_tau_days = 90.0 (используется только в weighted режиме)

Retrieval:
topk_brand = 4000
topk_seed = 4000
target_candidates = 10000 (после добивки популярностью)
max_candidates = 10000 (жёсткий потолок после сортировки/дедупа, до ранкера)

Популярность:
t_pop_days = 30
popularity_score(user) = tx_count_30d(user) * nunique_brand_dk_30d(user)

Ранкер:
catboost_loss = "YetiRank"
catboost_iterations, catboost_depth, catboost_learning_rate, catboost_l2_leaf_reg — задаются в params.yaml (дефолты должны быть выставлены в проекте и использоваться всегда, если не переопределены).
calibration_method = "platt_logreg_1d" (обучение LogisticRegression на одном признаке raw_score)

Reasons:
reasons_top_k = 3

NaN policy:
nan_policy = "keep" или "impute"
Если nan_policy="keep": оставляем NaN как есть.
Если nan_policy="impute": числовые NaN заполняем 0, категориальные NaN заполняем строкой "__MISSING__". Дополнительно добавляем бинарный флаг is_null_<col> для каждого столбца, где были NaN.

Разбиение на train/val (локальная оценка и подбор калибровки)
1) Собрать список всех offer_id из offer, отсортировать по offer.start_date по возрастанию.
2) split_idx = floor(0.8 * n_offers).
3) train_offers = первые split_idx офферов.
4) val_offers = оставшиеся офферы (последние 20% по start_date).
5) split_date = минимальная start_date среди val_offers (offer_sorted[split_idx].start_date).

Правило “снимка данных” для локальной валидации (чтобы избежать утечки)
Для построения всех обучающих артефактов (ALS, индекс похожих офферов, агрегации, train-датасет ранкера) используется только история транзакций/чеков до split_date:
- transaction_train = transaction[event_date < split_date]
- receipts_train = receipts[date_operated < split_date]
- offer_train_catalog = offer[offer_id in train_offers] (и связанные merchant)
Для построения val-датасета ранкера и предиктов (raw_score) также используется только transaction_train и receipts_train (без доступа к событиям >= split_date).
Ground truth для MAP@100 на валидации считается по полным транзакциям (полный transaction), но модель и фичи обязаны использовать только transaction_train/receipts_train.

Целевая переменная y (единственная и строгая)
Для пары (user_id, offer_id):
brand = brand_dk(offer)
window_start = offer.start_date
window_end = offer.end_date
y = 1 если:
- существует транзакция t пользователя с t.brand_dk == brand и window_start <= t.event_date <= window_end
- и не существует транзакции t0 пользователя с t0.brand_dk == brand и t0.event_date < window_start
Иначе y = 0.

Технически y строится отдельно для train и для val, но всегда по одному правилу; для локальной оценки MAP@100 используется полный transaction (чтобы получить “будущие транзакции”), однако признаки/кандидаты строятся на transaction_train.

Seed users для источника “similar offers → centroid”
Для каждого похожего оффера o_sim формируется множество seed_users_sim(o_sim) как объединение:
1) users из offer_activation для o_sim, где activation_date в [o_sim.start_date, o_sim.end_date]
2) users из offer_reward для o_sim, где event_date в [o_sim.start_date, o_sim.end_date]
3) users, у которых y(user, o_sim)=1 по правилу выше (через транзакции)

Итоговый seed_users для текущего оффера = объединение seed_users_sim по всем o_sim из top-K похожих офферов.
Для вычисления центроида используются только seed_users, присутствующие в ALS индексе (имеющие user_factors).
Если seed_users_count > 20000:
- считаем seed_strength(user) = число похожих офферов, в которых этот user попал в seed_users_sim
- сортируем seed_users по (seed_strength desc, user_id asc)
- оставляем первые 20000

ALS: построение и обучение (step 1)
Вход: transaction_train (или полный transaction для production).
1) Построить пары (user_id, brand_dk) и агрегировать их в счетчики count(user, brand).
2) Построить CSR матрицу R размера (n_users, n_brands):
- Индексация пользователей и брендов фиксируется через pandas.factorize на отсортированных уникальных значениях:
  user_ids_sorted = sorted(unique user_id)
  brand_ids_sorted = sorted(unique brand_dk)
  user_idx = позиция user_id в user_ids_sorted
  brand_idx = позиция brand_dk в brand_ids_sorted
- Режим als_mode:
  a) binary:
     значение R[u,b] = 1 если count(u,b) > 0, иначе 0
  b) weighted:
     для каждой транзакции вычислить w_tx = log1p(count(u,b)) * exp(-(ref_date - event_date)/als_time_decay_tau_days),
     где ref_date = max(event_date) по использованному набору транзакций,
     затем R[u,b] = сумма w_tx по всем транзакциям этого (u,b)
3) Обучить implicit ALS:
- model = AlternatingLeastSquares(factors=als_factors, iterations=als_iterations, regularization=als_regularization)
- Перед обучением, если используется implicit alpha-схема, применить confidence scaling: R_conf = als_alpha * R
- fit на CSR матрице R_conf
4) Получить factors:
- user_factors = model.user_factors (float32)
- brand_factors = model.item_factors (float32), где item соответствует brand_dk
5) Нормировать вектора для cosine retrieval:
- user_factors_norm[i] = user_factors[i] / max(||user_factors[i]||, 1e-12)
- brand_factors_norm[j] = brand_factors[j] / max(||brand_factors[j]||, 1e-12)
6) Сохранить артефакты:
- user_ids_sorted, brand_ids_sorted
- user_factors_norm, brand_factors_norm

Offer text embeddings + FAISS (step 2)
Вход: offer_train_catalog (для локальной валидации) или полный offer каталог (для production).
1) Для каждого offer_id взять offer_text; если NaN → заменить на пустую строку "".
2) Посчитать эмбеддинг через deepvk/USER-base:
- токенизация с max_length=offer_text_max_length, truncation=True, padding="max_length"
- mean pooling по last_hidden_state с учетом attention_mask
- L2-нормировка эмбеддинга
3) Построить FAISS индекс:
- IndexFlatIP по эмбеддингам (так как L2-нормировка → IP == cosine)
- Сохранить массив offer_ids_in_index в том же порядке, что и в матрице эмбеддингов
4) Поиск похожих офферов:
- при запросе offer_id:
  - получить его эмбеддинг (из кеша/матрицы)
  - faiss.search(top=similar_offers_k + 1)
  - удалить сам offer_id из результатов
  - взять первые similar_offers_k

Популярность (fallback source, step 3.3)
Вычисляется на transaction_train (локально) или на полном transaction (production), но всегда относительно даты inference:
1) Для локальной валидации, ref_pop_date = split_date.
2) Для production, ref_pop_date = max(event_date) текущей версии transaction.
3) Для каждого user:
- tx_count_30d = число транзакций с event_date в [ref_pop_date-30d, ref_pop_date)
- nunique_brand_dk_30d = число уникальных brand_dk в том же окне
- pop_score = tx_count_30d * nunique_brand_dk_30d
4) Отсортировать пользователей по (pop_score desc, tx_count_30d desc, user_id asc) — это фиксированный порядок.

Генерация кандидатов для одного оффера (step 3)
Вход: (merchant_id, offer_id, top_n).
Данные: offer/merchant для определения brand; ALS factors; FAISS индекс; популярность.
Выход: список candidate_users длиной ровно max_candidates (или меньше, если пользователей недостаточно), и retrieval-фичи для каждого кандидата.

Алгоритм:
0) Найти запись offer_id в offer. Если нет — 404.
1) Проверить, что offer.merchant_id_offer == merchant_id. Если нет — 404.
2) brand = brand_dk(offer) по merchant таблице. Если нет записи — 404.
3) Источник brand-sim:
- если brand присутствует в brand_ids_sorted:
  score_brand(u) = dot(user_factors_norm[u], brand_factors_norm[brand])
  взять topk_brand пользователей по score_brand (desc, tie by user_id asc)
  rank_brand начинается с 1
- иначе список пустой
4) Источник seed-centroid:
- найти similar_offers_k похожих офферов через FAISS (по offer_text)
- собрать seed_users по строгому определению (activation/reward в окне + y=1 по транзакциям)
- оставить только seed_users, присутствующие в ALS; при необходимости ограничить до 20000 по seed_strength
- если после фильтра seed_users пуст:
  seed-list пустой
- иначе:
  centroid = mean(user_factors_norm[seed_users]) и затем L2-нормировка центроида
  score_seed(u) = dot(user_factors_norm[u], centroid)
  взять topk_seed пользователей по score_seed (desc, tie by user_id asc)
  rank_seed начинается с 1
5) Начальный кандидатный сет:
cand_set = union(brand_topk, seed_topk)
6) Добивка популярностью:
Если |cand_set| < target_candidates:
- добавлять пользователей из popularity list в порядке убывания pop_score, пропуская уже имеющихся, пока |cand_set| == target_candidates или не кончатся пользователи.
Если |cand_set| >= target_candidates: добивка не выполняется.
7) Для каждого кандидата вычислить retrieval поля:
- rank_brand (если кандидат в brand_topk иначе 1e9)
- score_brand (если кандидат в brand_topk иначе 0.0)
- rank_seed (если кандидат в seed_topk иначе 1e9)
- score_seed (если кандидат в seed_topk иначе 0.0)
- rank_pop (позиция кандидата в popularity list начиная с 1, если был добавлен через popularity или вообще присутствует в списке; иначе 1e9)
- pop_score (если есть, иначе 0.0)
- retrieval_rank = min(rank_brand, rank_seed, rank_pop)
8) Фильтр current_client:
Удалить кандидата, если существует транзакция по brand с event_date < offer.start_date.
Эта проверка должна выполняться на том наборе транзакций, который доступен модели:
- локально: transaction_train
- production: полный transaction текущей версии
9) Если после фильтра кандидатов стало меньше target_candidates:
повторно пройти popularity list и добавлять новых пользователей, которые не current_client, пока не добьём до target_candidates или пока не исчерпаем пользователей.
10) Финальная сортировка и обрезка max_candidates:
Отсортировать кандидатов по:
- retrieval_rank asc
- max(score_brand, score_seed, pop_score_norm) desc, где pop_score_norm = pop_score / max_pop_score (0..1)
- user_id asc
Взять первые max_candidates.
Эта сортировка фиксирована и используется одинаково в train/val/inference.
11) Выход: candidate_users и их retrieval-фичи.

Построение признаков для ранкера (step 5)
Фичи строятся строго относительно offer.start_date (X1-A) и всегда используют только историю до start_date.
В локальной валидации используются transaction_train/receipts_train; в production используются полные transaction/receipts версии.

Опорная дата:
ref_date = offer.start_date.

Окна:
w ∈ {15, 30, 60, 90}
Окно считается как [ref_date - w дней, ref_date), то есть:
- события с date >= ref_date - w и date < ref_date
- события в ref_date (день старта) не включаются, чтобы избежать утечек по “внутри окна оффера”.

Статические user-фичи (из people/segments):
- age_bucket (cat)
- gender_cd (cat)
- region (cat)
- last_activity_day → преобразовать в days_since_last_activity = (ref_date - last_activity_day).days (numeric; если NaN → NaN или 0 по nan_policy)
- segment (cat)
- region_size (cat)
- auto, traveler, entrepreneur, vip_status (numeric 0/1)

Financial_account user-фичи:
На уровне user_id:
- fa_n_accounts_total = число строк
- fa_nunique_product_cd = число уникальных product_cd
- fa_product_mode = самый частый product_cd (cat)
- fa_product_mode_share = доля строк с этим product_cd
- fa_account_status_mode = самый частый account_status_cd (cat)
- fa_account_status_mode_share = доля
Даты open_month/close_month в ранкер напрямую не включать; при желании преобразовать в:
- fa_min_open_month, fa_max_open_month как числовые yyyymm (но только если не ломает nan_policy)

Транзакционные user-фичи по каждому окну w:
Для subset транзакций пользователя в окне w:
- transaction_count_w = число транзакций
- merchant_tx_count_w = число уникальных merchant_id_tx
- nunique_brand_dk_w = число уникальных brand_dk
- online_transaction_rate_w = доля online_transaction_flg == "Y" среди транзакций (если 0 транзакций → NaN)
- mostpop_amount_bucket_w = мода amount_bucket (cat)
- mostpop_amount_bucket_share_w = доля транзакций с mostpop_amount_bucket_w (numeric)
- brand_mode_w = мода brand_dk (cat)
- brand_mode_share_w = доля транзакций с brand_mode_w
- min_date_trans_w = min(event_date) в окне
- max_date_trans_w = max(event_date) в окне
- span_days_trans_w = (max_date_trans_w - min_date_trans_w).days (numeric; если <2 транзакций → NaN)
Дополнительно (для full history, без суффикса w):
те же метрики, но на истории до ref_date (event_date < ref_date).

Receipt user-фичи по каждому окну w:
Для subset receipts пользователя в окне w:
- receipt_count_w = число чеков (строк)
- log_receipt_count_w = log1p(receipt_count_w)
- active_days_receipt_w = число уникальных date_operated дней
- top_cat_receipt_w = мода category_name (cat)
- top_ratio_cat_receipt_w = доля чеков с этой category_name
- top_cost_receipt_w = мода items_cost (cat; items_cost — бакет)
- top_ratio_cost_receipt_w = доля чеков с этим items_cost
- min_date_receipt_w, max_date_receipt_w, span_days_receipt_w аналогично транзакциям
Дополнительно по истории до ref_date (без суффикса w) — те же метрики.

Offer/merchant фичи:
- offer_duration_days = (end_date - start_date).days (numeric; для 5999-01-01 будет большое число, оставляем как есть)
- offer_text_len = длина строки offer_text (numeric)
- offer_text_missing = 1 если offer_text == "" иначе 0
- brand_conflict_flg, n_merchants_for_brand (numeric)

Кросс-фичи из retrieval/ALS:
- dot_user_brand = dot(user_factors_norm[user], brand_factors_norm[brand]) если brand есть в ALS, иначе 0
- dot_user_centroid = score_seed из retrieval (если seed был, иначе 0)
- rank_brand, score_brand, rank_seed, score_seed, rank_pop, pop_score, retrieval_rank (все numeric; rank_*=1e9 если нет)

Список категориальных фичей для CatBoost (фиксированный)
Категориальными считаются и подаются в cat_features:
- age_bucket
- gender_cd
- region
- segment
- region_size
- mostpop_amount_bucket (и mostpop_amount_bucket_15d/30d/60d/90d)
- top_cat_receipt (и top_cat_receipt_15d/30d/60d/90d)
- top_cost_receipt (и top_cost_receipt_15d/30d/60d/90d)
- brand_mode (и brand_mode_15d/30d/60d/90d)
- fa_product_mode
- fa_account_status_mode
Все остальные — числовые.

Формирование train-датасета для CatBoostRanker (step 5 training)
1) Для каждого offer_id из train_offers:
- построить кандидатов ровно тем же алгоритмом, что и в inference, но:
  - FAISS индекс строится только по offer_train_catalog
  - seed_users берутся только из таблиц offer_activation/offer_reward и y по транзакциям, но все события берутся только из данных до split_date (как описано в “снимке”)
- получить candidates (до max_candidates) + retrieval-фичи
2) Для этих кандидатов построить все признаки относительно ref_date = offer.start_date, используя transaction_train/receipts_train.
3) Для каждого кандидата вычислить y по строгому правилу, но используя только transaction_train (так как обучаемся на снимке).
4) Если для offer_id число позитивов p == 0: этот offer_id пропускается (не добавляется в train), чтобы ранкер не получал группы без позитивов.
5) Негативное сэмплирование (строго):
- взять все позитивы (y=1)
- взять негативы (y=0) и отсортировать их по retrieval_rank asc, затем max(score_brand, score_seed, pop_score_norm) desc, затем user_id asc
- взять первые min(20 * p, count_negatives) негативов
- итоговый размер группы = p + выбранные негативы
6) Собрать общий train_df как конкатенацию групп. Сохранить столбцы:
- group_id = offer_id
- label = y
- все признаки
7) Обучить CatBoostRanker:
- loss_function = "YetiRank"
- random_seed = 42
- использовать cat_features из списка выше
- остальные гиперпараметры — строго из params.yaml
8) Сохранить модель в артефакты (например catboost.cbm) и список feature_names.

Формирование val-датасета и обучение калибровки (step 5 calibration)
1) Для каждого offer_id из val_offers:
- построить кандидатов тем же алгоритмом (используя только snapshot: transaction_train/receipts_train и offer_train_catalog для FAISS)
- построить признаки относительно offer.start_date на snapshot данных
- вычислить y_val по строгому правилу, но используя полный transaction (не snapshot), чтобы получить “будущие транзакции” для честной оценки
2) Прогнать CatBoostRanker на val_df и получить raw_score для каждой строки.
3) Для калибровки использовать пары (raw_score, y_val) по всем строкам val_df:
- обучить LogisticRegression на одном признаке X = raw_score.reshape(-1,1), y=y_val
- solver="lbfgs", C=1e6, max_iter=1000, random_state=42
- получить a = coef_[0], b = intercept_
4) Итоговый score для API:
score = 1 / (1 + exp(-(a * raw_score + b)))
5) Сохранить a,b вместе с моделью.

Локальная оценка MAP@100 (для контроля качества)
Для каждого offer_id из val_offers:
1) Построить кандидатов и признаки на snapshot данных.
2) Получить raw_score от ранкера, откалибровать в score.
3) Сформировать top_100 пользователей по score (desc, tie by user_id asc), предварительно убедившись, что current_client отфильтрован на snapshot данных.
4) Ground truth users для оффера:
- построить по строгому правилу y, но используя полный transaction (будущие транзакции)
- GT = {user | y(user, offer)=1}
5) Посчитать AP@100 и затем MAP@100 по всем офферам val_offers.

Inference / API поведение (step 6)
1) /lookalike:
- проверить top_n в диапазоне 1..1000; иначе 400
- проверить наличие offer_id и соответствие merchant_id; иначе 404
- сгенерировать кандидатов и признаки
- получить raw_score → score через калибровку
- отсортировать по score desc, затем user_id asc
- взять top_n
- audience_size = количество реально возвращённых (может быть меньше top_n)
2) reasons:
- взять матрицу признаков для top_n строк
- получить SHAP values через CatBoost (для ранкера)
- для каждого признака f посчитать:
  mean_abs_shap(f) = mean(|shap_f|) по top_n
  mean_shap(f) = mean(shap_f) по top_n
- выбрать top 3 признака по mean_abs_shap desc (tie by feature_name asc)
- Для категориальных признаков в ответе формировать строку feature как:
  "<col>=<mode_value_among_top_n>" (например segment=u_02)
  где mode_value_among_top_n — наиболее частое значение колонки среди top_n
- Для числовых признаков feature = "<col>"
- impact = mean_shap(f) (raw, со знаком)
3) /lookalike/batch:
- обрабатывает каждый request независимо тем же пайплайном, возвращает results массив

Требования к детерминизму
1) Везде фиксировать random_seed=42:
- numpy random
- python random
- catboost random_seed
- sklearn random_state
2) Все сортировки и tie-breaks фиксированы и включают user_id asc как последний ключ.
3) При обрезке seed_users до 20000 использовать seed_strength и user_id, без случайного сэмплинга.

Требования к прогресс-барам
Во всех долгих циклах (обучение ALS, вычисление эмбеддингов офферов, сбор train/val групп по офферам) использовать tqdm с информативным desc.

Что именно считается “готовым” результатом ML-части
1) Реализованы шаги:
- train artifacts: ALS factors + mapping, offer embeddings + FAISS, popularity list, feature builders, CatBoostRanker, calibration a,b
- inference: candidate generation + feature computation + ranker predict + calibration + SHAP reasons
2) Локальная валидация: MAP@100 на val_offers считается и логируется.
