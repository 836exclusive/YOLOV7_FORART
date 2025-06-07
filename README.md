speeshspeesh / Sergey Golovchan / 836 | rbdclan.moscow 

speesh.ru 

# YOLOv7-Buffed

Форк YOLOv7. Расширенный функционал. Детекция + трекинг. Без глянца. Для тех, кто в теме.

Предыстория

В 2023 году я открыл этот способ, пытался его пропушить и применять где только можно. В следующем году успешно защитил диплом бакалавра на эту тему. Спустя время начал находить неестественные попытки это повторить. Ребята делали это с помощью AE и собственноручно. Такой способ реализации нарушает всю мою идеологию, ибо я считаю что настоящий вебпанк не должен быть фейковым.

Не претендую на уникальность, но спустя год раскатываю это в сеть чтобы как можно больше людей знали как делать нужно.

Некоммерческое решение. Всё открыто, всё по делу.

## Особенности

* Реал-тайм трекинг и детекция
* Аудио сохраняется отдельно и возвращается обратно в видео
* Визуальные тюны:
  * размеры объектов
  * рамки с цветом или без
  * диагонали
  * угловые маркеры

## Быстрый старт

```bash
git clone https://github.com/your-username/yolov7-buffed.git
cd yolov7-buffed
pip install -r requirements.txt
```

Скачать веса:

* YOLOv7: [ссылка](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt)
* YOLOv7-w6: [ссылка](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt)

## Источники

* YOLOv7 — [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* SORT — [https://github.com/abewley/sort](https://github.com/abewley/sort)
* Object tracking fork — [https://github.com/haroonshakeel/yolov7-object-tracking](https://github.com/haroonshakeel/yolov7-object-tracking)

---

Код открыт. Использование только для некоммерческих задач.
