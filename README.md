# Классификатор женского/мужского голоса
## Постановка задачи
Изучить возможность построения бинарного классификатора звукового сигнала на женский или мужской голос. 
## Данные
Используется [dev-clean](https://www.openslr.org/resources/60/dev-clean.tar.gz) с [openslr](http://www.openslr.org/60/).
Чтобы скачать воспользуйтесь
```
curl https://www.openslr.org/resources/60/dev-clean.tar.gz --output tmp; tar -xf tmp;rm LibriTTS data; rm tmp
```
## Установка
Используется Python 3.7
Для установки необходимых зависимостей используйте
```
pip install -r requirements.txt
```
