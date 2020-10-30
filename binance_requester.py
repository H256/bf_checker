import hashlib
import hmac
import time
from datetime import datetime
from urllib import parse

import requests


class BinanceFuturesRequester:
    f_base_url = "https://fapi.binance.com/fapi/"
    s_base_url = "https://api.binance.com/"
    api_weight = 2400

    def __init__(self, api_key, api_secret):
        self.key = api_key
        self.secret = api_secret

    def prepare_request(self, query):
        # print(query)
        time_now = int(time.time() * 1000)
        # dict-update updates IN PLACE!
        if query is not None:
            merged_query = dict(query)
            merged_query.update(dict({'timestamp': time_now}))
        else:
            merged_query = dict({'timestamp': time_now})

        # merged_query['recvWindow'] = 5000
        querystring = parse.urlencode(merged_query)
        h_sig = hmac.new(self.secret.encode('utf-8'),querystring.encode('utf-8'),
                         hashlib.sha256).hexdigest()
        merged_query['signature'] = h_sig
        headers = self.get_header()
        return merged_query, headers

    def get_header(self):
        return {'X-MBX-APIKEY': self.key}

    def get_api_weight(self, response):
        weight = response.headers['X-MBX-USED-WEIGHT-1M']
        return int(weight) if weight is not None else 0

    def get_position_risc(self):
        params, headers = self.prepare_request(None)
        userdata = requests.get(
            "{}v2/positionRisk".format(self.f_base_url),
            params=params,
            headers=headers)
        weight = self.get_api_weight(userdata)

        return userdata, weight

    def get_balance(self):
        params, headers = self.prepare_request(None)
        userdata = requests.get(
            "{}v2/balance".format(self.f_base_url),
            params=params,
            headers=headers)
        weight = self.get_api_weight(userdata)

        return userdata, weight

    def get_income_data(self, income_type=None, limit=1000, start_time=None, end_time=None):
        p = {'limit': limit}
        if income_type is not None:
            p['incomeType'] = income_type
        if start_time is not None:
            p['startTime'] = start_time
        if end_time is not None:
            p['endTime'] = end_time
        params, headers = self.prepare_request(p)
        income_data = requests.get(
            "{}v1/income".format(self.f_base_url),
            params=params,
            headers=headers)
        weight = self.get_api_weight(income_data)

        return income_data, weight

    def transfer_amount(self, amount, asset='USDT', type=2):
        # types:
        # 1 => spot to usdt futures
        # 2 => usdt futures to spot
        # 3 => spot to coin futures
        # 4 => coin futures to spot
        if amount is None or amount == 0:
            return None, None
        p = {'asset': asset, 'amount': amount, 'type': type}
        params, headers = self.prepare_request(p)
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
        transfer = requests.post("{}sapi/v1/futures/transfer".format(self.s_base_url),
                                 data=parse.urlencode(params),
                                 headers=headers)
        return transfer
