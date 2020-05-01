from flask_restful import Resource, Api, reqparse
from text_segmenter import process_spine
import werkzeug
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
        audioFile = args['file'].read()
        return process_spine(audioFile)


api.add_resource(SpineOCR, '/')
