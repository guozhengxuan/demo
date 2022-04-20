# coding=utf-8
from flask import Flask, url_for, request, render_template
from flask_cors import *

# 配置浏览器跨域请求
app = Flask(__name__)
# cors = CORS(app, resources={r"/*": {"origins": "*"}})
# 中文编码
# app.config['JSON_AS_ASCII'] = False
