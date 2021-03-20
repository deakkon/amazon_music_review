import json

import tornado.ioloop
import tornado.web

from src.model.bertlike.annotator import BertlikeAnnotator


class BertHandler(tornado.web.RequestHandler):

    def initialize(self, model):
        self.pr = model

    def post(self):
        data = json.loads(self.request.body.decode('utf-8'))
        for k, v in data.items():
            try:
                assert k in ['summary', 'review']
            except AssertionError:
                raise tornado.web.HTTPError(status_code=400,
                                            log_message=f"Accepted keys of the JSON request are 'summary' and 'review'. You are passing {data.keys()}.")

        print('Got JSON data:', data)
        predictions = self.pr.get_predictions({
            'summary':data['summary'],
            'review':data['review']
        })
        self.finish({'someList': predictions})


def make_app():
    return tornado.web.Application([
        (r"/", BertHandler, dict(model=BertlikeAnnotator())),
    ], debug=True, autoreload=True)

if __name__ == "__main__":
    app = make_app()
    app.listen(5555)
    tornado.ioloop.IOLoop.current().start()
