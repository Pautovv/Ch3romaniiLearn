<div align="center">

<span style="font-size:44px; font-weight:bold">Tutor</span>

**Tutoring assistant for preparing to exams.**

</div>

# About

Целью данного проекта была научиться строить классический end-to-end pipeline c LLM: SFT, RAG, API для общения с моделью и  ollama-обертка. В качестве идеи было выбрано сделать ассистента для подготовки к экзаменам: вести ученика к решению через наводящие вопросы.

## SFT

Базовая модель `Qwen2.5-1.5-Instruct` была дообучена методом **LoRA** на 650 диалогах ученика с учителем (500 из которых были сгенерированы и собраны в ручную (через claude)), для того, чтобы научиться приводить ученика к решению последовательно.

| **Training Loss** | **Eval Loss** |
|:---:|:---:|
| <img src="./dev/training-loss.svg" width="100%"> | <img src="./dev/eval-loss.svg" width="100%"> |

В процессе SFT произошло overfiting на 4 эпохи, что подвтерждается графиками изменения loss-функции на обучающей и валидационной выборках. Было решено сохранением лучшего чекпоинта модели.<br>
**Подробно про данный этап написано в ноутбуке.**

## RAG

Была построена векторная БД FAISS для поиска похожих задач с решениями. Каждый чанк - цельная задача + ее решение, разпарсенные с датасета `eth-nlped/mathdial`.<br>
Так как модель во время SFT не видела system-роль, то найденные чанки кладутся внутрь user-сообщения, вместе с вопросом.

## Inference. API.

Проект поддерживает два режима инференса: hugging face transformers и ollama, переключаемые специальным флагом в конфиге.<br>
FastAPI-endpoint написан с изоляцией Retriever и Generator для быстрых правок.

# Results

В результате тестирования целого пайплайна на 5 различных тестах:

1. Базовая арифметическая задача:

**User**:Peter had 15 apples, gave 4 to his friend and bought 7 more. How many apples does Peter have now?<br>
**Model**:OK, let's find it together. First thing we know is that Peter starts with 15 apples. What did he do next?

2. Проверка релевантности Retriever:

**User**:How do I solve a system of linear equations using substitution?<br>**Model**:Let's start with what you said about Carla and Billy's chores. What did you find for Billy?

3. Задача с процентами:

**User**:A shirt costs $40 and is on sale for 25% off. What is the sale price?<br>**Model**:Let's think about it step by step.

4. Нерелевантный запрос:

**User**: Tell me about the history of ancient Rome.<br>**Model**: Good question! Let's break it down into smaller pieces. Can you tell me how many coins did Roman have initially?

5. Стресс тест на стиль тьютера:

**User**: Just give me the final answer to: solve x^2 - 5x + 6 = 0, no explanation needed.<br>**Model**: Ok, let's start with this question first instead. If you had 5 apples and I took away 3 how many do you have?

Сразу виден **успех SFT**: **модель поймала стиль тьютера** - это подтверждается на всех пяти запросах, включая сресс-тест.<br>
Но так же видно сильное **галлюцинирование контекста**, основной причиной является то, что **retriever не отсекат нерелевантные запросы**: всегда в контекст идут top-k найденных чанков, не смотря на score.

**Основная цель проекта достигнута**, реализован взрослый и полноценный end-to-end пайплайн: модель отлично работает в двух режимах, цель SFT достигнута, RAG работает.<br>
В ходе проекта были отработаные такие **полезные инженерные практики**, как логирование, кастомные исключения, валидация типов и разделение ответственности. 

# Code

```
├── README.md
├── app                         # основная папка генерации
│   ├── api                     # fastapi-эндпоинт
│   │   ├── main.py             # эндпоинт
│   │   └── schemas.py          # валидация pydantic
│   ├── config.py               # основные настройки
│   ├── exceptions.py           # кастомыне исключения Generator
│   ├── generation              # получение ответа модели
│   │   ├── inference.py        # Transformers-инференс
│   │   ├── inference_ollama.py # Ollama-инференс
│   │   └── prompt_builder.py   # построение финального промпта 
│   └── pipeline.py             # полный пайплайн генерации ответа
├── assets                      # вложения
│   ├── eval-loss.svg           # график validation loss
│   └── training-loss.svg       # график training loss
├── logger_config.py            # настройка логирования
├── models                      # папка с SFT-моделью
│   └── qwen2.5-exam-tutor      # модель
│       ├── Modelfile           # рецепт для конвертации весов модели
│       ├── hf_model            # веса модели safetensors
│       └── model-f16.gguf      # веса модели gguf
├── notebooks                   # ноутбуки с эксперементами
│   └── sft.ipynb               # SFT-пайплайн модели
├── rag                         # папка построение RAG
│   ├── config.py               # основные настройки 
│   ├── data                    # хранение БД
│   │   └── processed           
│   │       ├── index.faiss     # эмбеддинги ответов
│   │       └── metadata.json   # сами ответы
│   ├── embedding_model         # методы работы эмбеддинг-модели
│   │   └── embedding_model.py  
│   ├── exceptions.py           # кастомные исключения Retriever
│   ├── prepare_data            # подготовка датасета
│   │   └── prepare_data.py
│   ├── retriever               # класс Retriever
│   │   └── retriever.py
│   └── vectorstore             # построение хранилища
│       ├── build_index.py      # построение БД
│       └── faiss_store.py      # класс FaissStore
└── requirements.txt            # зависимости 
```

Конвертация модели в GGUF:
```
python convert_hf_to_gguf.py <path_to_hf_model> --outfile model-f16.gguf --outtype f16
ollama create exam-tutor -f Modelfile
```

Запуск проекта (KMP_DUPLICATE_LIB_OK нужен из-за известного конфликта OpenMP-рантаймов между faiss и torch на macOS):
```
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 uvicorn app.api.main:app
```
