from flask_restful import Resource, Api, reqparse
from SpineClassifier.text_segmenter import process_spine
from SpineClassifier.book_spine_extractor import SpineExtractor
import werkzeug
import json
from flask import Flask

app = Flask(__name__)
api = Api(app)


class SpineOCR(Resource):
    def get(self):
        return {'spine': 'ocr'}

    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        image_file = args['file'].read()
        return process_spine(image_file)

class ShelfOCR(Resource):
    def get(self):
        return json.loads('books.json')
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        image_file = args['file'].read()
        json.dumps(SpineExtractor.extract_spines_from_img_str(image_file))

api.add_resource(SpineOCR, '/')
api.add_resource(ShelfOCR, '/shelf/')
