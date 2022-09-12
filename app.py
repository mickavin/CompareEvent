import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from flask import Flask, request, jsonify, abort, make_response
from flask_restful import Resource, Api
import pandas as pd 
from math import isnan
from flask_cors import CORS

from zipfile import ZipFile
import pathlib
import io
import numpy as np
import datetime

import requests
import socket
import requests.packages.urllib3.util.connection as urllib3_cn

from urllib.parse import urlparse #, unquote, parse_qs
import urllib.request

__version__ = "0.3.0"
app = Flask(__name__)
cors = CORS(app, resources={r"*": { "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Origin"]}}, allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Origin"])
api=Api(app)

PORT = os.environ.get("PORT")

dtypes_datafeed = {
    "Tickets:venue_address": str,
    "aw_deep_link": str,
    "aw_image_url": str,
    "aw_product_id": int,
    "brand_name": str,
    "custom_1": str,
    "custom_2": str,
    "custom_3": str,
    "custom_4": str,
    "custom_5": str,
    "custom_6": str,
    "custom_7": str,
    "custom_8": str,
    "custom_9": str,
    "description": str,
    "merchant_image_url": str,
    "merchant_name": str,
    "product_name": str,
    "promotional_text": str,
    "search_price": float,
    "merchant_product_category_path": str,
    "valid_to": str,
    "Tickets:longitude": float,
    "Tickets:latitude": float,
    "product_model": str,
    "savings_percent": str,
    "merchant_product_second_category": str,
    "Tickets:primary_artist": str
}

def isNaN(value):
    try:
        return isnan(float(value))
    except:
        return True



def get_paginated_list(results, url, start, limit, related=None):
    start = int(start)
    limit = int(limit)
    count = len(results)
    if count < start or limit < 0:
        abort(404)
    # make response
    obj = {}
    obj['start'] = start
    obj['limit'] = limit
    obj['count'] = count
    # make URLs
    # make previous url
    if start == 1:
        obj['previous'] = ''
    else:
        start_copy = max(1, start - limit)
        limit_copy = start - 1
        obj['previous'] = url + '?start=%d&limit=%d' % (start_copy, limit_copy)
    # make next url
    if start + limit > count:
        obj['next'] = ''
    else:
        start_copy = start + limit
        obj['next'] = url + '?start=%d&limit=%d' % (start_copy, limit)
    # finally extract result according to bounds
    obj['results'] = results[(start - 1):(start - 1 + limit)]
    if related is not None:
        obj['related'] = related.to_dict('records')
    return obj

def allowed_gai_family():
    """
     https://github.com/shazow/urllib3/blob/master/urllib3/util/connection.py
    """
    family = socket.AF_INET
    #if urllib3_cn.HAS_IPV6:
    #    family = socket.AF_INET6 # force ipv6 only if it is available
    return family

def downloadFile():
    url = "https://productdata.awin.com/datafeed/download/apikey/d325af8704887d91048306bf04ea5892/language/fr/fid/23455,48133,57151/columns/aw_deep_link,product_name,aw_product_id,merchant_product_id,merchant_image_url,description,product_model,merchant_category,search_price,Tickets%3Alongitude,Tickets%3Avenue_name,Tickets%3Alatitude,merchant_product_category_path,merchant_name,aw_image_url,Tickets%3Avenue_address,brand_name,product_short_description,valid_to,Tickets%3Aevent_date,Tickets%3Aprimary_artist,promotional_text,custom_1,custom_3,custom_5,custom_7,custom_9,custom_2,custom_4,custom_6,custom_8,savings_percent,merchant_product_second_category/format/csv/delimiter/%2C/compression/zip/adultcontent/1/"
    urllib.request.urlretrieve(url, "datafeed_552325.zip")
    url2 = "https://productdata.awin.com/datafeed/download/apikey/d67bacb1c5f6fee13322d0671d3a847c/language/fr/fid/22739/columns/aw_deep_link,product_name,aw_product_id,merchant_product_id,merchant_image_url,description,search_price,merchant_category,merchant_name,aw_image_url,brand_name,product_short_description,product_model,promotional_text,merchant_product_category_path,valid_to,custom_1,custom_2,custom_3,custom_4,custom_5,custom_6,custom_7,custom_8,custom_9,Tickets%3Aevent_date,Tickets%3Aprimary_artist,Tickets%3Avenue_name,Tickets%3Avenue_address,Tickets%3Alongitude,Tickets%3Alatitude,savings_percent,merchant_product_second_category/format/csv/delimiter/%2C/compression/zip/adultcontent/1/"
    urllib.request.urlretrieve(url2, "datafeed_1005721.zip")
    
def unzipFile():
    with ZipFile("datafeed_552325.zip","r") as zip_ref:
        zip_ref.extractall()


    
def parse_date(data):
    cr_date = data
    cr_date = datetime.datetime.strptime(cr_date, "%Y-%m-%dT%H:%M:%S")
    cr_date = cr_date.strftime("%d/%m/%Y %H:%M")
    return cr_date


class Events(Resource):
    def __init__(self):
        data = pd.read_csv("datafeed_552325.zip", compression='zip', header=0, sep=',', quotechar='"', dtype=dtypes_datafeed, encoding = "utf-8")
        
        self.data = data.to_dict('records')
    
    def post(self):
        response = make_response(jsonify(get_paginated_list(
        self.data, 
        '/events', 
        start=request.args.get('start', 1), 
        limit=request.args.get('limit', 20)
        )))
        return response

class search(Resource):
    def __init__(self):
        self.data = pd.read_csv("datafeed_552325.zip", compression='zip', header=0, sep=',', quotechar='"', dtype=dtypes_datafeed, encoding = "utf-8")
        urllib3_cn.allowed_gai_family = allowed_gai_family
    
    def haversine_vectorize(self, lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        newlon = lon2 - lon1
        newlat = lat2 - lat1

        haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon/2.0)**2

        dist = 2 * np.arcsin(np.sqrt(haver_formula ))
        km = 6367 * dist
        return km
    
    def post(self):
        name = ""
        city = ""
        merchant = ""
        type = ""
        category = ""
        start_date = 0
        end_date = 0
        latitude = None
        longitude = None
        distance = None
        price = []
        bl = True 
        isApp = False
        isApp = request.args.get('isApp', False)
        id = request.args.get('id', None)
        relatedItems = None
        if(request.json is not None):
            name = request.json.get('name', "")
            city = request.json.get('city', "")
            merchant = request.json.get('merchant', "")
            type = request.json.get('type', "")
            category = request.json.get('category', "")
            start_date = request.json.get('startDate', 0)
            end_date = request.json.get('endDate', 0)
            latitude = request.json.get('latitude', None)
            longitude = request.json.get('longitude', None)
            distance = request.json.get('distance', None)
            id = request.json.get('id', None)
            price = request.json.get('price', [])
            isApp = request.json.get('isApp', False)
        merchant = request.args.get('merchant', "")
        # find data from csv 
        if len(merchant) > 0 :
            self.data = self.data[self.data['merchant_name'].str.contains(merchant, case=False, na=False)]
        if len(name) > 0 :
            self.data = self.data[(self.data['product_name'].str.contains(name, case=False, na=False)) | (self.data['custom_9'].str.contains(name, case=False, na=False)) | (self.data['custom_8'].str.contains(name, case=False, na=False))]
        if len(city) > 0 :
            self.data = self.data[(self.data['custom_7'].str.contains(city, case=False, na=False)) | (self.data['promotional_text'].str.contains(city, case=False, na=False)) | (self.data['Tickets:venue_address'].str.contains(city, case=False, na=False))]
        if len(type) > 0 :
            self.data = self.data[(self.data['custom_6'].str.contains(type, case=False, na=False)) | (self.data['custom_1'].str.contains(type, case=False, na=False))]
        if len(category) > 0 :
            self.data = self.data[(self.data['merchant_product_category_path'].str.contains(category, case=False, na=False))]
        if start_date > 0 and end_date == 0:
            print("first", datetime.datetime.fromtimestamp(start_date))
            self.data = self.data[(self.data['merchant_name'] == "Fnac Spectacles FR") & (datetime.datetime.fromtimestamp(start_date) < pd.to_datetime(self.data['valid_to'].str.split(pat=";").str[0], errors='coerce', format='%Y-%m-%d'))]
            print("second", self.data)
        if start_date == 0 and end_date > 0:
            self.data = self.data[(self.data['merchant_name'] == "Fnac Spectacles FR") & (datetime.datetime.fromtimestamp(end_date) > pd.to_datetime(self.data['Tickets:event_date'].str.split(pat=";").str[0], errors='coerce', format='%Y-%m-%d'))]
        if start_date > 0 and end_date > 0:
            self.data = self.data[(self.data['merchant_name'] == "Fnac Spectacles FR") & (datetime.datetime.fromtimestamp(end_date) > pd.to_datetime(self.data['Tickets:event_date'].str.split(pat=";").str[0], errors='coerce', format='%Y-%m-%d') | datetime.datetime.fromtimestamp(start_date) < pd.to_datetime(self.data['valid_to'].str.split(pat=";").str[0], errors='coerce', format='%Y-%m-%d'))]
        if (latitude is not None) and (longitude is not None) and (distance is not None):
            self.data["distance"] = self.haversine_vectorize(self.data["Tickets:longitude"], self.data["Tickets:latitude"], longitude, latitude)
            self.data = self.data[(self.data['distance'] < distance)]
        if len(price) == 2 :
            self.data = self.data[((self.data["search_price"] <= price[1]) & (self.data["search_price"] >= price[0]))]
        if id is not None:
            r = self.data[(self.data["aw_product_id"] == int(id))]
            a = r["Tickets:event_date"].str.split(pat=";").str[0]
            r["datetime"] = pd.to_datetime(a, errors='coerce', format='%Y-%m-%d', utc=True)
            ref = r.to_dict('records')
            if len(ref) > 0:
                if pd.isnull(ref[0]["datetime"]):
                    ref[0]["datetime"] = "NAN" 
                relatedItems = self.data[(self.data["aw_deep_link"] != ref[0]["aw_deep_link"]) & (self.data["Tickets:venue_address"] == ref[0]["Tickets:venue_address"]) & (self.data["merchant_product_second_category"] == ref[0]["merchant_product_second_category"])].head(12)
                self.data = self.data[(self.data["api_city"] != "NAN") & (((self.data["api_name"] != "NAN") & (self.data["api_name"].str.upper() == ref[0]["api_name"].upper())) | (self.data["product_name"].str.upper() == ref[0]["product_name"].upper())) & (self.data['api_city'].str.upper().str.replace("/", " SUR ", regex=False).str.contains(ref[0]["api_city"].upper(), case=False, na=False)) & (((self.data["merchant_name"] == "Fnac Spectacles FR") & (self.data["Tickets:event_date"] == ref[0]["Tickets:event_date"])) | ((self.data["merchant_name"] == "Carrefour Spectacles FR") & (self.data["custom_8"] == ref[0]["custom_1"])) | ((self.data["merchant_name"] == "CDiscount Billetterie FR") & (pd.to_datetime(self.data["custom_2"].str.split(pat=";").str[0], errors='coerce', format="%Y-%m-%dT%H:%M:%S", utc=True) == ref[0]["datetime"])) | ((self.data["merchant_name"] == "See tickets FR") & (pd.to_datetime(self.data["Tickets:event_date"].str.split(pat=";").str[0], errors='coerce', format="%Y-%m-%d %H:%M:%S", utc=True) == ref[0]["datetime"])))]
                if len(self.data.to_dict('records')) == 0:
                    self.data = r
            else:
                self.data = r
                relatedItems = self.data[(self.data["aw_deep_link"] != ref[0]["aw_deep_link"]) & (self.data["Tickets:venue_address"] == ref[0]["Tickets:venue_address"]) & (self.data["merchant_product_second_category"] == ref[0]["merchant_product_second_category"])].head(12)
        # return data found in csv
        response = make_response(jsonify(get_paginated_list(
        self.data.to_dict('records'), 
        '/search', 
        start=request.args.get('start', 1), 
        limit=request.args.get('limit', 20),
        related=relatedItems
        )))       
        return response
    
class changeFile(Resource):
    def __init__(self, country: str = "FR", unique: bool = True):
        downloadFile()
        dat = pd.read_csv('datafeed_552325.zip', compression='zip', header=0, sep=',', quotechar='"', dtype=dtypes_datafeed)
        dat2 = pd.read_csv("datafeed_1005721.zip", compression='zip', header=0, sep=',', quotechar='"', dtype=dtypes_datafeed, encoding = "utf-8")
        self.df = pd.concat([dat, dat2])

    def searchCity(self):
        return self.df[['merchant_name',"promotional_text", "product_short_description", "Tickets:venue_address"]].apply(lambda row: str(row["promotional_text"]) if row['merchant_name'] == "Carrefour Spectacles FR" else (str(row["Tickets:venue_address"]) if row['merchant_name'] == "Fnac Spectacles FR" else str(row["Tickets:venue_address"]) if row['merchant_name'] == "CDiscount Billetterie FR" else str(row["Tickets:venue_address"])),axis=1)

    def searchName(self):
        return self.df[['merchant_name',"custom_9", "custom_8","product_name"]].apply(lambda row:  str(row['custom_9']).upper() if row['merchant_name'] == "Carrefour Spectacles FR" else (str(row['custom_8']).upper() if row['merchant_name'] == "Fnac Spectacles FR" else str(row['product_name']).upper()), axis=1)
    
    def post(self):
        self.df['api_city'] = self.searchCity()
        self.df['api_name'] = self.searchName()
        compression_opts = dict(method='zip', archive_name='datafeed_552325.csv')  
        self.df.to_csv('datafeed_552325.zip', index=False,compression=compression_opts) 

        return make_response(jsonify({'message': "success"}))

api.add_resource(Events, '/events')
api.add_resource(search, '/search')
api.add_resource(changeFile, '/reload')

if __name__ == "__main__":
    app.run()