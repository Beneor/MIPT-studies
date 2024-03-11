Программа Sokrat-bot-generative используется для генерации ответов на запрос пользователя ("Сократ") в соответствии с текстами "Диалоги" Платона.
Реплика представляет собой ответ собеседника Сократа, сгенерированный в соответствии с промптном "Ответ на вопрос".
Модель основана на предобученной модели TheBloke/Llama-2-7B-Chat-fp16.
График обучения и примеры ответов даны в ноутбуке Sokrat_bot_generative.ipynb
Датасет подготовлен с использованием ноутбука Plato-dataset.ipynb.
Исходные тексты диалогов содержатся в папке Plato_dialogs.
Все необходимые файлы и каталоги содержатся в папке sokrat_bot_generative.
Для обучения модели следует использовать ноутбук Sokrat_bot_generative.ipynb (в Сolab). При этом необходимо указать путь к каталогу sokrat_bot_generative.
Обученную модель адаптеров следует поместить в папку sokrat_bot_generative/checkpoint-500 
Обученную модель адаптеров можно скачать отсюда:
https://drive.google.com/drive/folders/12nguZDKExLzxrJkEX0Em0MkI97H7g8gY?usp=sharing

В качестве веб-сервиса использована программа Gradio. Для работы с веб-сервисом следует ввести веб-адрес, выданный программой, в строку браузера. 

Для запуска программы можно использовать ноутбук Sokrat-inference.ipynb или скрипт sokrat-inference.py, предварительно установив все необходимые библиотеки.

Requirements: Colab, GPU.
Библиотеки, необходимые для работы:
accelerate, bitsandbytes, faiss-gpu, gc, gradio, matplotlib, numpy, pandas, random, torch, tqdm, trl, transformers.

!pip install -q -U bitsandbytes
!pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install accelerate
!pip install faiss-gpu
!pip install trl==0.4.7
!pip install gradio

