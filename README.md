
# Book Spine Classifier
This application provides a flask RESTful API to detect and classify books in a bookshelf

## Prerequisites
to install this on your own server

install requirements via pip
> pip install -r requirements.txt

install Tesseract OCR engine via thier website or apt get the apps in the Aptfile and set teh respath to tesseract and the tessdata in your environment

## Built with
* Python
* Pycharm - IDE
* OpenCV - computer vision framework
* TesseractOCR - text recognition engine
* Heroku - deployment enviroment

## Download
You can download a flutter(android/IOS) application that interfaces with this API from the following repository [Book Classifier App](https://github.com/rameziophobia/book_classifier_flutter_app)

## Usage
You can
* use the application referenced above
* Post http request to the hosted API on heroku
    * post a single book spine on '/'
        >  example using curl: curl -v -F "file=@img_path.jpg" https://book-spine.herokuapp.com/
    * post a shelf book spine on '/shelf/'
        >  example using curl: curl -v -F "file=@img_path.jpg" https://book-spine.herokuapp.com/shelf/

## Authors

* **Ramez Noshy** - *Main Developer* - [rameziophobia](https://github.com/rameziophobia)
* **Pierre Adel** - *Main Developer* - [PierreAdel](https://github.com/PierreAdel)

## License
this project is licensed under the MIT license
