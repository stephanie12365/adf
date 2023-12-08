
from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
import insightface
import shutil
import os
from os import listdir
from os.path import isfile, join
from flask import jsonify
import tempfile
from datetime import datetime
import random
from flask import session
from collections import defaultdict
from pathlib import Path
import string
import secrets
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


app = Flask(__name__)
codigouser = None



a = 1


app.secret_key = secrets.token_hex(16)
print("insightface", insightface.__version__)


# Define la ruta al directorio de modelos buffalo_l
model_dir = os.path.join('archivos', '.insightface', 'models', 'buffalo_l')
os.environ['INSIGHTFACE_HOME'] = model_dir
model_dir = os.path.join('archivos')
# Crea una instancia de FaceAnalysis y evita la descarga automática del modelo
app_insightface = insightface.app.FaceAnalysis(name="buffalo_l", model_dir=model_dir, download=False, download_zip=False)
app_insightface.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model("inswapper_128.onnx", download=False, download_zip=False)


def generarnumero():
    global codigouser
    
    # Generar 3 números aleatorios
    numeros = ''.join(random.choices(string.digits, k=3))

    # Generar 4 letras aleatorias
    letras = ''.join(random.choices(string.ascii_uppercase, k=4))

    # Mezclar números y letras
    codigouser = ''.join(random.sample(numeros + letras, k=7))
    print ("CODIGOUSER", codigouser)
    

    return codigouser


#############################################################
    
# Diccionario para almacenar el número de veces que un código ha sido validado
codigos_validos = {

  'RjtFcpUA': 0, 'YBrnkOHl': 0, 'CGVynMZ4': 0, 'fb5SiQbD': 0, 'cTGw4EMd': 0, "tefiteamo":-1000,
  '2L1dnLef': 0, 'kHoUiBed': 0, 'AInuhwpu': 0, 'D77jdrmz': 0, 'gEcRlYlT': 0, 'S7nzKkVJ': 0, 'mXtj7fB3': 0, 'yu63nx8D': 0, 'P0elYQSJ': 0, '786q3bo1': 0, 'StHp2Kk0': 0, 'v42RFd42': 0, 'uKHlRhkx': 0, '0RQM5ygn': 0, 'ZxEFa9Ws': 0, 'tuH6O4Yr': 0, 'NtAL5ZmB': 0, '55MsweXe': 0, 'aIbkRVxs': 0, 'NOBBKeQE': 0, 'FCvW1246': 0, 'TOX3wesC': 0, 'Dfp90bCy': 0, 'DIoWyAmg': 0, 'PAzDfOlS': 0, 'Du4S7DOh': 0, 'ewF4x8tk': 0, '7iSoLWXP': 0, '2rLkvNZA': 0, '3vKxOxrB': 0, 'YjSb2aDe': 0, 'z8G1mBMN': 0, 'MADA7Ri7': 0, 'sBEgWgtQ': 0, 'pLueis4f': 0, 'ODYWWk78': 0, 'Z8kq1TTe': 0, 'BkokVzPj': 0, 'OewffCH7': 0, 'PrzNSwvz': 0, 'pTAICrw6': 0, 'szoyR5Vb': 0, 'T7ZZyK2p': 0, 'M2uE7RXX': 0, 'T7W1qZuA': 0, 'BJqzPiXJ': 0, 'Gs0fFpFb': 0, '7GTNp9GY': 0, 'apohUHP5': 0, 'Y2JgUCUA': 0, 'xhWBRQUu': 0, 'dZWXF1KF': 0, '31Reg4YY': 0, 'KZlVeOF4': 0, '4F7GE9Hz': 0, 'RCVmo5kp': 0, '8ZW1227u': 0, 'xpNv3ND2': 0, 'SiFbjGjf': 0, 'p4pZ57Ht': 0, 'qh55WNbP': 0, 'gpAsNzY7': 0, 'e6lrhFFo': 0, 'y4DM2lfp': 0, 'WKVRh87v': 0, 'DF2WwYiB': 0, 'c6JSzIwT': 0, 'yWwgPUe5': 0, 'S1PdZFmg': 0, 'MuiK6H2l': 0, 'pkqSFtyV': 0, 'C8m9uJ5q': 0, 'GsnICDKX': 0, 'OFmTMwzn': 0, '7Ewh5L1q': 0, 'PwaqszDb': 0, '0YWKtASR': 0, 'JONATm9D': 0, 'Z7n2AfR4': 0, 'cSZL94ee': 0, 'xU0GNfcs': 0, 'xuQaodNx': 0, 'FQFVPEae': 0, 'U8wEg70x': 0, 'z6TREyiD': 0, '9GrlBKeD': 0, 'clkMV3W2': 0, 'S6coUIYQ': 0, 'TLDZfoMy': 0, 'qffNYGOl': 0, 'zlbqsMmI': 0, 'nCGiaWXI': 0, '4yqjxmZf': 0, 'tCQDk6C7': 0, 'TpZlBRaX': 0, 'zrmDjCRK': 0, 'JvQcwsbP': 0, 'ZLBzyMjH': 0, 'A8XIHZMR': 0, 'BOIpiVwz': 0,
  'JwVhCEZt': 0, 'rhNhrVF4': 0, 'HV5jRGfh': 0, 'fWTu7pKY': 0, 'VeFjErF4': 0, 'pbRCy9dK': 0, 'xuDd1Gd5': 0, 'u1JQ46r4': 0, 'xyha5tmn': 0, 'JF4Jjezk': 0, 'A7aV3sMW': 0, 'wQqS7WkW': 0, 'fZPhfrX7': 0, 'by9ShSvm': 0, 'pycUcsfg': 0, 'kzThUfCg': 0, 'Fn1sjHFG': 0, 'TXUgbdER': 0, 'DcGGgb63': 0, '9cZrwwbg': 0, 'UK9bnHCY': 0, 'xCwGa2K7': 0, 'NAtmedUA': 0, '45b241Y4': 0, 'ykauPTpD': 0, 'xj9Em4cg': 0, 'DfQenQxP': 0, 'pKgvfjXz': 0, '3CjYYqGX': 0, 'eJp37cTC': 0, 'TFKzZRwu': 0, 'Fyxvz8HT': 0, '4aZFxjhq': 0, 'Em8RhAhA': 0, '3kf6bd9v': 0, 'df5rKRf8': 0, '4F35mR1W': 0, 'Xxpgshk2': 0, '3UGT5Ka2': 0, 'bYUmWh21': 0, 'sRQCKw1Q': 0, '5jfkxNQp': 0, '5AXQk67W': 0, 'zPu3D5cD': 0, 'm1qVvFKW': 0, 'xhZD9pJm': 0, 'dzSMEhah': 0, 'mSyKcFJ3': 0, 'UMkC88Vk': 0, 'xvp9VxrH': 0, 'STfG6xt5': 0, '27FP3mTQ': 0, 'CxY7cMnQ': 
0, 'tnumtZuD': 0, 'ffXzSSsz': 0, 'EY23Qpn9': 0, '8Z8qjmBf': 0, 'xgHTgay4': 0, 'a7u4dyt7': 0, 'QsjcwMsj': 0, 'SEcqCpKy': 0, 'xDxsEyND': 0, 'jeSZUCuX': 0, 'yea16xtW': 0, 'gMDZx2cP': 0, 'RVd2gvqS': 0, 
'6N3u59vW': 0, 'gWAX8Xew': 0, 'UDmJRTRX': 0, 'k7AHqkQt': 0, 'gZtQZdBS': 0, 'G3cT25yS': 0, 'ccpUeyRU': 0, 'xvSKT15d': 0, '8au8GHr2': 0, 'Kh1DEMSS': 0, 'whtS2esh': 0, 'xD5tjn9Q': 0, 'yDeXtZqz': 0, 'Jya115Cu': 0, '5t7E42bK': 0, 'BQunRDSd': 0, 'ptgVV7su': 0, 'eHpCxVE8': 0, 'u9GUnru7': 0, 'Ga2rTJUJ': 0, 'H1aSFFmd': 0, 'J8sAWW4s': 0, 'gXtw9V4A': 0, '6S8VgbA6': 0, 'RwsZrSBT': 0, '1CJ6nudp': 0, 'j1EUPUvD': 0, '6JgSQUcx': 0, 'arqYGQd5': 0, 'fxWgUU2c': 0, 'yzkE31fB': 0, 'aYk6PAs2': 0, '8psupDwf': 0, 'mdYrCN9d': 0, 'pAygBmm2': 0, 'BkQZW5cs': 0, 'pZcZ3PVY': 0, 'UfFnkabk': 0, 'kkZFY4w8': 0, 'hsgpwuGA': 0, 'xgY28Vwg': 0, 'HqFZbREJ': 0, 'jVaZTDas': 0, 'BbhVqFrB': 0, 'QAYwnYJg': 0, 'beSQdzdb': 0, '4jUaExEe': 0, 'hKp5HKJa': 0, '2P4jcZk7': 0, 'VrvCpdJB': 0, 'arRz2RuV': 0, 'NJXZ8EfW': 0, 'qpnHTS62': 
0, 'mhRw5V3p': 0, 'EeqtDQPf': 0, '4zsMU1Px': 0, 'wEBKjtyP': 0, 'RRJhxAZ3': 0, 'PRqnwgnh': 0, 'p121CExs': 0, 'nBW2m7FQ': 0, 'ah7a6ym8': 0, '8AcFrYXx': 0, 'XSsyA36b': 0, 'w467qN5n': 0, 'YBQv1KyS': 0, 
'QVzz4zTk': 0, 'StdAa5aU': 0, 'sukj4BCd': 0, 'HpyYRuhd': 0, 'fEbYgsV7': 0, 'Rd8upVZw': 0, '9CwZ3Wdr': 0, 'etuzeHry': 0, '3rP7j4Xy': 0, 'rK7KvCN2': 0, 'Rpm8vvmN': 0, 'CXRRkuQw': 0, 'TnCfA7yd': 0, 'QNMkq87y': 0, '9VKgBgke': 0, 'mKyzMrW8': 0, 'UFKVdgEU': 0, 'Fb27nHBk': 0, '1u8AY8W6': 0, 'vC4qCTkv': 0, 'nVkN1uXQ': 0, 'BaAznhy7': 0, 'K5DfAtSs': 0, 'MVjh71sv': 0, 'Nw11J28E': 0, 'DyvTcUQT': 0, 'bU8sq63V': 0, 'vPAnAb8T': 0, 'AvXPMAVE': 0, '8MBRFJ9n': 0, 'zJeTDz4S': 0, 'dX5rm9YP': 0, '3r2wHjf7': 0, 'wHWjydna': 0, 'pUbmQhqg': 0, '8dx9GY9T': 0, '4hbrBaH6': 0, 'CsV86FSq': 0, '9ZwRpCV6': 0, 'UQb7TRAh': 0, 'XQxyyYU4': 0, 'd3JtNYcF': 0, '6xNJhEEe': 0, 'a9p7aaJJ': 0, 'yW634s3s': 0, '8mQbDNZV': 0, 'TxGh2M5S': 0, 'VKv4rFKd': 0, 'k2dFG6yb': 0, 'X1n7yeUc': 0, 'brVG2adr': 0, '5PvdSQj3': 0, 'WHRYTSpk': 
0, 'nSvnCBCk': 0, 'Jpg3A4XR': 0, 'DgubyfPx': 0, 'rtktXTPJ': 0, 'gSz4aY97': 0, 'fw3CWhaF': 0, 'XZZFYTFu': 0, 'hwTdPc3r': 0, 'GCApTUDP': 0, 'R6C9mTxq': 0, 'X9p5Hx8m': 0, 'YF13nW44': 0, 'J8Fh75Dq': 0, 
'3T4yfDN1': 0, '1g1EDzW1': 0,'xm2RA4Yx': 0, 'FSy2RV7E': 0, 'hc1qUDhr': 0, 'eyh7fdmz': 0, 'X5Z3dyhE': 0, 'zJ96R9yp': 0, '5nCkn4aW': 0, 'EZsjrP6N': 0, 'XJfBQkC9': 0, 'AcdGzqSw': 0, 'AgaskvpY': 0, 'yeAxKk7v': 0, 'RBQ1gr2c': 0, 'm22pBMv5': 0, 'ZayeQ81D': 0, 'kxGvHkZp': 0, 'n2w7wC6E': 0, 'et2cyQ9a': 0, 'uwfu5D4p': 0, 'MGuMgwnT': 0, 'BFf3MKGX': 0, 'ne4qy2gy': 0, 'dFZEVS7j': 0, 'fSvqakQH': 0, 'nUxvfUDA': 0, 'fearynFw': 0, 'UZU1BW4p': 0, 'xPgWjmKY': 0, 'Pc9FQE1k': 0, 'XKx4SkTM': 0, 'cGGYSNd5': 0, 'hk8P6se6': 0, 'v4UDxgz6': 0, 'ryfKxtFb': 0, 'adYmjGnq': 0, 'FgYxw7R6': 0, 'ypVUrAmn': 0, 'jmcGw9v9': 0, 'xJMVYcyD': 0, 'DuG2ZkYR': 0, 'PvANhx86': 0, 'vndmcm6m': 0, 'hxQsUzUa': 0, 'PkmcMudj': 0, 'HED19zxj': 0, '4f7fAbgm': 0, 'tt1DgjVM': 0, '8vPQGSVj': 0, '4Uv9pKzr': 0, 'zaNQR1gR': 0, 'jUdgbN8u': 0, 'jB3TkG2r': 0, 'w7VGvbwz': 
0, 'gC5VMhrT': 0, 'RjfTWZb7': 0, 'd1Rr2nKx': 0, 'u8GH3hGH': 0, 'bhhbw45G': 0, '96YxczKk': 0, '9tXHV5aT': 0, 'aD3KmA3h': 0, '3ksrfEtt': 0, 'vEdsYE3m': 0, 'R2gJ6xfG': 0, 'wAXMxkhm': 0, 'TBqMVztE': 0, 
'ZDv7GTbw': 0, 'yjZ9CHW3': 0, 'eNBBUD5e': 0, 'btKH36mj': 0, 'KHuKCv7V': 0, 'aRHV5R3E': 0, 'NJFpCpgQ': 0, '3yNAbSTj': 0, 'rP5qYRbg': 0, 'w2cXbDX6': 0, 'sMMAr7BP': 0, '84tgGK3A': 0, 'nu869MRZ': 0, 'h2pNsvFn': 0, 'm2sT9Aer': 0, 'dWfbFYwr': 0, 'kfw1Rv4k': 0, 'kqTHw4jn': 0, '7mCnzezd': 0, 'tVZjSwD9': 0, 'fqqwBYda': 0, 'EmQ5eGRN': 0, 'F11HTbsa': 0, '9sXqjqxY': 0, 'UvyKTJFB': 0, 'ecjcXsp4': 0, 'uS33ZBaJ': 0, 'wZTxZHE3': 0, 'JeHg1Zet': 0, 'SANEhFE1': 0, '7pH82mKf': 0, '8CQDYPQ4': 0, 'C5PZ5Rh6': 0, 'GmYE5kFU': 0, 'C64rVCKc': 0, 'NTM4Tun2': 0, '8g1naAS5': 0, 'ZEDgDJ9W': 0, 'VsAtjA2X': 0, 'AbsYPfRz': 0, 'CHU2URcd': 0, 'XUdSDxqW': 0, '3CbSEKj2': 0, 'J168Z44J': 0, '4GdN3Ev3': 0, 'vTmaHVEQ': 0, 'K88ucVZa': 0, 'BfP91Haj': 0, '2mdAwmNY': 0, 'ZwAjUGw9': 0, 'g6Ee3WpD': 0, 'aXCBJYh1': 0, 'KMgrN7Pg': 
0, 'HEvsxpeD': 0, '5EE9xbN4': 0, 'FygHZtPz': 0, 'UA5jTyb7': 0, 'bXk9FhhG': 0, '2v7jC4VP': 0, 'WRfdepJs': 0, 'Phk6mnp2': 0, '8NUESbFy': 0, 'pc98p1uN': 0, 'a4WASU5Z': 0, 's73yNu2Y': 0, 'YV11SatH': 0, 
'gjN5an33': 0, 'XbzKNZhD': 0, 'gGQPFZyC': 0, '4n3sjBHM': 0, 'jgFGdGeG': 0, 'SRG7ESkB': 0, 'qHCt4hns': 0, '8kA3rW5e': 0, 'qR6GcmqM': 0, '35tHjecs': 0, 'h4KfRmE6': 0, '2V43VXgA': 0, 'YHzzWRED': 0, '3abvwfRb': 0, 'CyPZGbPJ': 0, 'WUUR5T3J': 0, 'H3XeZPz7': 0, 'WNUjbmRu': 0, 'zjHsG2Rm': 0, 'bzTPWfe7': 0, 'yPfcpd71': 0, 'nKFp9NAP': 0, 'AXyKqq8B': 0, 'tk5KyTHY': 0, 'nJwHCVWm': 0, 'bcbtP3GF': 0, 'vNZgf2Gf': 0, 'QU19jfJ7': 0, 'dvXfDK28': 0, 'ZugBTbfS': 0, 'AjzhCWQG': 0, 'cgF7gmvJ': 0, 'CgCZXFW7': 0, 'NqYweQwN': 0, 'RjqH51Aj': 0, 'x3BV5SyZ': 0, '142bEVA9': 0, 'NhWpt44H': 0, 'NR9HrHmF': 0, 'nwVXCjK9': 0, 'vGTbz95Q': 0, 'HYXs2Qcc': 0, 'mxfNy8gg': 0, 'vqyaCFXp': 0, 'rJe1UA7X': 0, 'S1ugC36u': 0, 'rEqkXAUk': 0, 'w31tS8rp': 0, 'pU6QBKv2': 0, 'kRrd3sNC': 0, 'fZq4qWvD': 0, 'YYNyKDRq': 0, 'bnjjyzNK': 
0, 'SR4QH1UX': 0, 'H9XHjbWv': 0, 'EjuBaSMn': 0, 'g75Qc6D7': 0, 'KwgfJ9jE': 0, 'AJXxSFTG': 0, 'CXFhsZKW': 0, 'fd9tf3hv': 0, '1J4mUgrG': 0, 'UVcRN7Dw': 0, '3UhPhgen': 0, '17y3GrKf': 0, 'GByhy3bY': 0, 
'zZ2hcKpV': 0, 'BuP1p2BE': 0, 'XrR7V4uD': 0, 'axAA9eGe': 0, 'RGjP4V19': 0, 'QWeCpKtN': 0, 'U3w1rzMK': 0, 'D5MB4JZ4': 0, 'fxjGReNN': 0, '26DfjRND': 0, 'NkfMDbsA': 0, 'TnA15x84': 0, 'F5zz7UTb': 0, '8wbhAdzp': 0, 'S5xQccJV': 0, 'McuSdmss': 0, 'vVB4ANnK': 0, 'f6aFjaqB': 0, 'tZj6yys5': 0, 'cp1V8SYc': 0, 'XJ6NANQS': 0, '19zNzXzz': 0, 'VY5JPaR8': 0, 'hFSeSBG4': 0, 'pbXZYzsF': 0, 'jy9E9W8E': 0, 'qkxKb4bW': 0, '5zgvPZ8Q': 0, 'QHBwAYpf': 0, 'cuyMSQSp': 0, 'NX18veVd': 0, 'TuvRctKp': 0, 'AqpfQgvV': 0, 'PfQfE45N': 0, '3JQN96PP': 0, 'AQ6hJeqz': 0, 'YrHYA3va': 0, '6r9kba8N': 0, 'd7pVEMxs': 0, 'ykjUzfkX': 0, 'X7CsBSCJ': 0, 'SSHh8nzr': 0, 'ECFjD53e': 0, '7c3S55Vn': 0, 'S5xA6YwK': 0, 'dDC2rTsj': 0, 'tAM96nzX': 0, 'mtqnzevP': 0, 'NNsKRdsN': 0, 'VEzzPWjd': 0, 'g4BpsgSy': 0, 'KTtSV7NQ': 0, 'vkRvThEy': 
0, 'vhWWZGj7': 0, 'bx1Bzdpv': 0, 'EYJpXXqq': 0, '3XmhjpKe': 0, 'qPAY9f4d': 0, 'q6bsCqY6': 0, 'YMazyJpA': 0, 'hq2wKVgd': 0, '3yPb8Mn5': 0, 'XFa8r49q': 0, 'YTsPWc8X': 0, '7eMTwb1t': 0, 't6Pxa9JX': 0, 
'hARpcSZz': 0, 'BGFMM3fM': 0, 'SXnyjTKY': 0, 'EwdW5gqR': 0, 'UDQ2ypKz': 0, 'ctxKwxJZ': 0, 'WEbGHrWd': 0, 'kePYtQTu': 0, '53QZAvh3': 0, 'eff1rmGZ': 0, 'EqmHTadB': 0, 'XgGSZFkx': 0, '6P8mmJ1R': 0, 'wGXX7u84': 0, 'BFCdqVyW': 0, 'hVnWAX3Z': 0, 'WnKB36WQ': 0, 'd5BJ9ZMk': 0, '3ZPrkYCH': 0, '6qZYu7ur': 0, 'gG4NSrqu': 0, 'YQaSFDkN': 0, '8TQjP4Kr': 0, '1NUsrFMx': 0, '1MajgCAT': 0, 'fAdxhH4g': 0, 'sEZCuWKA': 0, 'TyHnEadv': 0, 'jMsA65dZ': 0, '1xuzjk3U': 0, 'ggKk26Eh': 0, 'JyrSPtha': 0, 'Zcbgehg1': 0, 'cAGcTvEc': 0, 'SFbhasD8': 0, 'fNgDwUKa': 0, 'bt3fmjSf': 0, '8dwbf27f': 0, '2vDX9P1k': 0, 'yCvfwkEW': 0, 'BVaQSfPJ': 0, 'Um1Xr9Yh': 0, '9j3Ae9RX': 0, 'aG5t63hd': 0, 'khSCZ6CE': 0, 'dksRMRKX': 0, 'u3pSjnQK': 0, '8NPQ77nq': 0, 'SAnpyHp1': 0, 'cvkP4YUt': 0, 'xyjjHN1q': 0, 'ZFuQjMST': 0, 'ccJX1wgs': 
0, '8bFVPV9j': 0, 'Zx8JpxHv': 0, 'cXcmTaTT': 0, 'C9f8Y5eG': 0, 'pUAmQTCc': 0, 'Spp9nHMV': 0, 'vj6NJe92': 0, 'JNxNp7aD': 0, 'AEqbmZNH': 0, 'qF4wSU2S': 0, 'Aj3rGTMM': 0, 'ZaP7Sfdg': 0, '98axCEGa': 0, 
'Mt6KRu7d': 0, 'JamSSGT9': 0, 'DbU7uf2h': 0, 'UEMV2XzW': 0, 'EkBBXKeW': 0, 'jV3V4Vp8': 0, 'Gy8JfpJ9': 0, 'hKG6YHcJ': 0, 'GNpEbM3h': 0, 'SA5wwgYw': 0, '83x2fNmt': 0, '57cGBu2v': 0, 'BBdFKSk5': 0, 'zAx4RRcf': 0, 'TUXQ2McC': 0, 'PwHbK93s': 0, '3M6ZYZzz': 0, 'VdxUgwCq': 0, 'jeFZFGAf': 0, 'ftVGeWqp': 0, '4AsTuj15': 0, 'B5scBbrK': 0, 'mW73fm5y': 0, 'cy2sw3b4': 0, 'bUbhfFCu': 0, 'E4PZG6cf': 0, 'bUkMapCv': 0, 'wU1XBy92': 0, 'RyttKzmH': 0, 'hmA3jCpm': 0, 'cGjQX9NU': 0, '4K4TVzh5': 0, 'XzzKEmgH': 0, 'fffKkKuM': 0, 'arxzZCdS': 0, 'EPaV6zum': 0, 'a59YXHuG': 0, 'nHASD5nV': 0, '66gG3uWN': 0, 'D3N6hqG5': 0, 'mMDYCa1A': 0, '7TC384bE': 0, '6X1jZhub': 0, 'XXBtSMak': 0, '73P81brj': 0, 'Zec6v8QE': 0, 'f8jvQ5Ru': 0, '1EJDqqTK': 0, 'fznffnKm': 0, 'wtt7XXka': 0, 'qEFSFGNd': 0, '2eGVdhmT': 0, 'YBFHR6xY': 
0, '6UUhkRVh': 0, 'tT7JWRn2': 0, 'vr9swQaF': 0, 'SRTq7rgZ': 0, 'fFzPXBuG': 0, 'w5nHyJF4': 0, 'JBqfFMau': 0, 'jwRut2EH': 0, 'hn14Q1up': 0, 'D9Tm9Ybk': 0, 'NxzZvzhb': 0, 'HqtyVSXH': 0, 'YjVB58S4': 0, 
'wMa7HGGh': 0, 'TarYECUf': 0, 'j2jF1PsX': 0, 'GWUWhQQ5': 0,
'7u41pbtb': 0, '2jcsvhzs': 0, 'bkswpdsh': 0, '2gz3zjgf': 0, 'gj2rj4hd': 0, 'f7una7r3': 0, 'wudgca9g': 0, 'fjwkwr9t': 0, '34tavqtw': 0, 'uqqqb1fn': 0, 'c63sy4b6': 0, 'rwbee5m3': 0, 'wne1xkjk': 0, 'ts3f4nza': 0, 'p1a6mwu1': 0, 'ash8mt4v': 0, '4evgwy1w': 0, '1xcrjpnd': 0, '36vw34wk': 0, 'umy7q436': 0, 'nbh55pyu': 0, '6e3rrpmj': 0, 'xjbek8pm': 0, 'rjdghemc': 0, 'pnaswt7x': 0, 'g7rzhxcx': 0, '9unzqkun': 0, 'y8hy1zmf': 0, 'xdjb7hnt': 0, 'zuehr43v': 0, 'h8jrk8np': 0, 'fa6grgm3': 0, 'wga7ykyh': 0, 'rxgagbqu': 0, '7jt4tvhz': 0, 'hnhez4dw': 0, 'nkcfksag': 0, 'w9e45tbm': 0, 'xn8pb7km': 0, 'nx8u4ex3': 0, 'yukwwjg1': 0, 'vcjd6btk': 0, 'rmbg43vz': 0, 'yw6zbqnb': 0, 'f2yv8134': 0, 'wwvvwj17': 0, 'krjg2rpr': 0, 'kqc3n4ay': 0, 'an635ttx': 0, '2guen8hm': 0, 'qvw2bvax': 0, 'u8j8xrs2': 0, 'vcwm4832': 
0, 'z813u7xh': 0, '592rthxu': 0, 'pexm31en': 0, 'g2jd5jts': 0, 'cnn5427w': 0, 'h22wgx3h': 0, 'zmwj3j3v': 0, 'qgru9xb5': 0, '6rya5bgm': 0, 'kweucned': 0, 'h2dd4zdq': 0, 'swa6n84t': 0, '2wy4tqyr': 0, 
'p89twfjn': 0, 'c1ny1j1n': 0, 'h7cp8r14': 0, '1we4qdwf': 0, 'wkgz9583': 0, '6mwdw6me': 0, 'b4qvt1r7': 0, 'bzr4j5vq': 0, 'b16t9cry': 0, 'm54g2s3m': 0, '7q6dtx76': 0, 'q77ycn1u': 0, 'r2nksvmm': 0, 'gjn6ngdb': 0, 'zxbjftfv': 0, '6ewkuqrt': 0, 'rkgnusb4': 0, '2q8nmy2w': 0, 'esuq9p91': 0, 'y9pdexjg': 0, 'kb3nugd9': 0, 'q427enj8': 0, 'gh519g29': 0, 'u5wucrb9': 0, 'ymprj63k': 0, 'ww5a532e': 0, 'ptfa6aq8': 0, '92m6fbat': 0, 'ccamwfqd': 0, 'sfraz7yn': 0, 'k641gqmu': 0, '54drj6rg': 0, 'jbrdjagz': 0, 'xpfe6hcd': 0, '2a9veucz': 0, 'hz3ed9mx': 0, 'rd6pyj5c': 0, 'dcdtrsrv': 0, '6spfyp5h': 0, 'auuwd5hc': 0, 'jv8ywcgj': 0, 'zgq38gzw': 0, 'kt7g37mj': 0, 'pf68d1n8': 0, 'ehk8pt4x': 0, '2d9ha9cq': 0, '7f33ptkg': 0, '4f31xt7d': 0, 'n1hves21': 0, 'dpfzbund': 0, 'akdg42f1': 0, 'uzsza866': 0, 'rac4eykj': 
0, 'zvm9tyy8': 0, 'jry4qtm8': 0, 'nqbmuahq': 0, '7mg64xcp': 0, 'sw5y24zy': 0, 'eapsvftf': 0, 'hfgdpyeb': 0, '3udea4re': 0, 'ejs2e6ex': 0, 'mzk4jye8': 0, '9tdygeqc': 0, 'ffvnwuxj': 0, '3an4tx43': 0, 
'whwwykyy': 0, '1q8bd78f': 0, 'xd8dtrg2': 0, 'gqk49n3g': 0, 'v11hen8s': 0, '1sg8jqgx': 0, 'aagcb27q': 0, 'mr6fj1s4': 0, '4c53pdw1': 0, '9ps4nut5': 0, 'b72ykyp8': 0, 'rgfx3rn1': 0, 'dkbycabz': 0, 'dk34jp3v': 0, 'z43fv8wy': 0, '81htwk7d': 0, '65xywppv': 0, 'k6q547jp': 0, 'k1x8c44d': 0, '8gqvsf1e': 0, 'agndstrv': 0, '9svr9cmu': 0, 'vupf9ecv': 0, '5kufnch6': 0, 'sv2ek298': 0, 'edq2zzh9': 0, 'chpf59zk': 0, 'v6h3qpsw': 0, 'hj2jzky9': 0, 'b21pdqjv': 0, 'r2tyd7yc': 0, '3pndgba6': 0, 'cc6ucmve': 0, '3684p971': 0, '19exy1bs': 0, 'tb9nf6y2': 0, '4hugdvy5': 0, 'rf6ajf3x': 0, '9r4j5cgb': 0, 'ftrw849c': 0, 'tfmhptrj': 0, 'dj2em147': 0, 'j3w1yaee': 0, 'hfcr2nx2': 0, '7kmryt3f': 0, 'n7w6kf1a': 0, 'yb8dpn5k': 0, '5p7n8fct': 0, 'a8hxe1r7': 0, '8rz1j7uj': 0, 'mb4h18fx': 0, '821kww7m': 0, 'r85k66gp': 
0, 'j89k5h4p': 0, 'dwhkgjrm': 0, 'qjprmnf8': 0, 'gbhdjr52': 0, 'w8fagmrz': 0, 'djsdnzea': 0, 'k9gdt3ay': 0, 'zdf2jbyu': 0, '4rpmrvg2': 0, 'tdwspx2w': 0, 'ng68bpyq': 0, '4dkpvqg8': 0, '228tjzkk': 0, 
's714ck5d': 0, 'hkafpa53': 0, 'nm8kfn9z': 0, 'g2kp5ubr': 0, 'nnmvf2yk': 0, 'kjq86rbh': 0, 'dgcnwezv': 0, '9e9yt24f': 0, 'c83zsf5s': 0, '7n2fnqay': 0, 'aczmkf53': 0, 't5bmnqzz': 0, 'z6v7pxb4': 0, '8b5xsthd': 0, 'vcsa2pzm': 0, 'kcp84cp5': 0, '839wdswa': 0, 'r8km8ysn': 0, 'sxr3mngw': 0, 'gacdc5zf': 0, 'c4rs1xbb': 0, '12rehjbj': 0, 'qghkj4fv': 0, '7z8wg3xz': 0, '2f8ec3wh': 0, 'vpqcbfyf': 0, 'sxmge53n': 0, 'c9ra7957': 0, 'sqm2ajcb': 0, '4j4hwr7c': 0, '5zf2k8fh': 0, 'wvuw7e46': 0, 'x4prhgxs': 0, 'byjfu1v9': 0, 'cw11ru39': 0, '2mem3rkm': 0, 'jt2fqabs': 0, 'pq147pfc': 0, 'gxqpp3bs': 0, 'w8mmz6ed': 0, 'aayzdsp4': 0, 'kqu8mrp6': 0, 'dxseph25': 0, '1uqp58ja': 0, 'hmj36rsy': 0, 'wvvnm7v8': 0, '912utf8g': 0, 'u9sgv56f': 0, 'ezk45y47': 0, 'rtntjfta': 0, '6tu9ju99': 0, '86u1vcce': 0, 'q6scuqnt': 
0, '42j8qqk9': 0, '5gpjrcuq': 0, 'nzepgkff': 0, '61esv4mv': 0, 'cvnxy5x5': 0, '4magg35b': 0, 'axcjnwc1': 0, '9jzwcf7q': 0, '67jm9bu7': 0, 'hf7efacp': 0, 'uwu4js48': 0, '2cbhtfv4': 0, 'pwfsg7n2': 0, 
'f4ffemnn': 0, 'q1rrex72': 0, '6ygq5vve': 0, '49hug2y9': 0, 'ayzz1e89': 0, 'eh652mhz': 0, '77mjrx8t': 0, '1mks4mbz': 0, 'fxzh53z4': 0, '21qcqvhr': 0, '25p7dkg4': 0, 'wr5d54qt': 0, 'dvrkye3g': 0, 'zvux5rmj': 0, 'pv4g514s': 0, 'axu8x9dh': 0, 'cen4dty8': 0, '7qebdznz': 0, 'zkrmfdgq': 0, 'jp3yckgt': 0, '7fpuapm3': 0, 'uspqym5j': 0, 'uytb1zyf': 0, 'ecv7h6yk': 0, 'yn3bhrf6': 0, '8neacfgh': 0, '66sdhc28': 0, 'vf4ave2a': 0, 'ezgn8t29': 0, 'c4g3wz45': 0, 'yv5pcmrd': 0, 'hnceyzhg': 0, 'eevambwt': 0, 'frtrjj8s': 0, 'qfjyue1k': 0, '7kq4e5zk': 0, '6fz6tphk': 0, '9xqvq5ug': 0, 'rkx85k6z': 0, '9vhcdsdk': 0, 'zzy12xxc': 0, 'yvehcf9m': 0, 'nh99mm81': 0, '1uje3kj5': 0, '5kr1pum5': 0, 'vu24k946': 0, 'f7nvqc1n': 0, 'txz9w5fp': 0, 'g2wx98ft': 0, '699p43uq': 0, 'zb1bqkq8': 0, 'hkb4z68h': 0, 'km22j4tj': 
0, '2n3yfw5n': 0, 'uhwadavk': 0, 'cfc25qvm': 0, 'mgae725c': 0, 't5x31c2x': 0, 'aeret3sw': 0, 'c8m33k1h': 0, '1uufp4br': 0, 'gqerg1gm': 0, 'byry43yh': 0, 'pu8xmpc2': 0, 'uzenze59': 0, 'uptjcq98': 0, 
'ttvwr7vy': 0, 'y8j9ygqj': 0, '292yu363': 0, 'tw21wr1a': 0, '8zxkjk4p': 0, '8pn8ydnp': 0, '2vbsku6x': 0, 'jt2zn3fw': 0, '6g5bdute': 0, '32uz4g6e': 0, 'q4jw81v3': 0, '2zfgb4ux': 0, '39tjecpx': 0, '88ypqp1j': 0, '6brvzmqn': 0, '6qh6a6wx': 0, 'dqvm89vw': 0, '5y8mq5ge': 0, '2r1c9mzc': 0, 'kkqx8xk7': 0, 'fq1bstmq': 0, 'n7y771pt': 0, 'nqvhj5wn': 0, 'wbvh6mss': 0, 'bcejtb8j': 0, '28vs9a1c': 0, 'u1q69ykj': 0, '4kupjmf7': 0, '433gwhj1': 0, 'zea77egh': 0, 'd5x2a8ns': 0, 'k83v5eyr': 0, 'sv2nctu2': 0, '6x248sgp': 0, 'n6wzms65': 0, '12gnfp5m': 0, 'eb8kkxy3': 0, 'zbhaz1nx': 0, 'yx35b5gx': 0, '8t5cfmd1': 0, '2s3qvxxr': 0, 'qdwvq87h': 0, 'pebuxp6z': 0, 'bag9p92a': 0, '4fc38rtr': 0, 'qgjp4eee': 0, 'xhjeg799': 0, 'chyvr2a1': 0, 'yj9r9jgy': 0, '4mnfy8v6': 0, 'p7g1q9tw': 0, 'vnemjyf3': 0, '4e7nug7h': 
0, 'cn4c4cf1': 0, '5ga4quz4': 0, 'bp5br95r': 0, 'wxrs729h': 0, 'nkpszhd3': 0, 'p2neftjf': 0, 'umvm6dw3': 0, 'fq5879xa': 0, 'jp9yqdxe': 0, 'xpvye7fz': 0, '25rkxzw8': 0, 'tnug4jer': 0, 'bws6hssn': 0, 
'tcemn9ej': 0, 'nkk5npqn': 0, '74gdsevw': 0, 'y895v34n': 0,
  }


limite_validaciones = 26  # Establece el límite de validaciones permitidas

# Diccionario para almacenar la lista negra de códigos
lista_negra = defaultdict(int)

@app.route('/validar_codigo/<codigo>', methods=['GET'])
def validar_codigo(codigo):
    global codigos_validos, lista_negra

    codigouser = generarnumero()
    print ("MI CODIGOUSER ES", codigouser)
    session['user'] = codigouser
    session['codigouser'] = codigouser
    if os.path.join('uploads', codigouser):
        # Si el directorio existe, lo eliminamos.
        try:
            shutil.rmtree(os.path.join('uploads', codigouser))
            print(f"Carpeta eliminada exitosamente.")
        except OSError as e:
            print(f"No se pudo eliminar la carpeta . Error: {e}")
    else:
        print(f"La carpeta {os.path.join('uploads', codigouser)} no existe aun.")
    if codigo in codigos_validos:
        if codigos_validos[codigo] < limite_validaciones:
            # Incrementa el contador
            codigos_validos[codigo] += 1
            # Imprime el estado actual de la cuenta
            print(f"Código {codigo} validado {codigos_validos[codigo]} veces.")
            # Devuelve éxito
            return jsonify({'status': 'success', 'message': 'Código válido'})
        else:
            # Excede el límite de validaciones
            if codigo not in lista_negra:
                lista_negra[codigo] = 1
                # Imprime el estado actual de la cuenta
                print(f"Código {codigo} agregado a la lista negra.")
            return jsonify({'status': 'error', 'message': 'Código ha alcanzado el límite de validaciones'})
    else:
        # Código no válido
        return jsonify({'status': 'error', 'message': 'Código no válido'})






##########################################################################

def rename_images():
    
    
    codigouser = session['codigouser']
    # Obtén la lista de archivos en la carpeta
    ip_folder_path = os.path.join("uploads", codigouser)
    ip_folder_path = session['ip_folder_path']
    files = os.listdir(ip_folder_path)

    # Filtra solo los archivos de imagen (puedes ajustar las extensiones según tus necesidades)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Ordena los archivos por fecha de creación
    image_files.sort(key=lambda x: os.path.getctime(os.path.join(ip_folder_path, x)))

    # Recorre la lista de archivos y renombra cada uno con un número
    for i, file_name in enumerate(image_files, start=1):
        file_path = os.path.join(ip_folder_path, file_name)
        new_name = f"{i}{Path(file_name).suffix}"  # Nuevo nombre con número de orden y extensión
        new_path = os.path.join(ip_folder_path, new_name)
        os.rename(file_path, new_path)
        






@app.route('/static_images')
def static_images():
    estilos_path = os.path.join(app.static_folder, 'estilos')
    files = [f for f in os.listdir(estilos_path) if os.path.isfile(os.path.join(estilos_path, f))]
    return jsonify({'files': files})

def construir_imfondo(imagefilename):
    b = session["b"]
    static_dir = os.path.join('static')
    # Construir la ruta de la imagen de fondo en el directorio 'grandes'
    imfondo_path = os.path.join(static_dir, b, 'grandes', imagefilename)
    session ["imfondo_path"] = imfondo_path
    # Verificar si el archivo existe
    if not os.path.exists(imfondo_path):
        print(f"El archivo no existe en la ruta: {imfondo_path}")
        # Puedes manejar el error de alguna manera, por ejemplo, retornar None
        return None

    return imfondo_path



@app.route('/select_image', methods=['POST'])
def select_image():
    
    
    codigouser = session['codigouser']
    if 'user' in session:
    
        if 'file' not in request.files:
            return jsonify({'error': 'No se encontró ningún archivo'}), 400

        file = request.files['file']

        # 1. Obtener user del cliente
        user_address = codigouser

        # 2. Crear una subcarpeta con el nombre de la dirección IP si no existe
        ip_folder_path = os.path.join("uploads", user_address)
        session['ip_folder_path'] = ip_folder_path
        if not os.path.exists(ip_folder_path):
            os.makedirs(ip_folder_path)

        # 3. Guardar la imagen en la subcarpeta
        img_persona_path = os.path.join(ip_folder_path, file.filename)
        file.save(img_persona_path)
        print("SE GUARDÓ EN", img_persona_path)
        session['img_persona_path'] = img_persona_path
    
        return img_persona_path
    else:
        # El usuario no ha iniciado sesión, redirige al formulario de inicio de sesión
        return redirect(url_for('seleccion_estilo'))



@app.route('/check_a')
def check_a():
    global a
    if a == 1:
        return jsonify({'a': 1})
        
    else:
        return jsonify({'a': 0})
    

@app.route('/')
def seleccion():
    
    return redirect(url_for('seleccion_estilo'))








@app.route('/imagen_final', methods=['GET'])
def imagen_final():
    global a
    a = 1
    
    codigouser = session['codigouser']

    # Obtén la dirección IP del usuario
    user_ip = codigouser

    # Obtiene la lista de archivos en el directorio 'static'
    static_dir = 'static'
    files = [f for f in listdir(static_dir) if isfile(join(static_dir, f))]

    # Filtra los archivos que contienen el valor de 'user_ip'
    filtered_files = [f for f in files if user_ip in f]

    # Ordena la lista de archivos por fecha de modificación
    filtered_files.sort(key=lambda x: os.path.getmtime(os.path.join(static_dir, x)))

    # Toma el archivo más reciente de la lista (si hay alguno)
    if filtered_files:
        latest_file = filtered_files[-1]
        result_image_name = latest_file  # Solo el nombre del archivo
        result_image = os.path.join(static_dir, latest_file)
        return render_template('imagen_final.html', result_image=result_image_name)
    else:
        return "ERROR: Los rostros no fueron cargados o no esperaste a que el sistema los cargue - Recuerda que antes de seleccionar un diseño debes cargar correctamente las caras de las personas tal como se indica en el video de inicio de la plataforma! Si no sigues los pasos detallados no obtendrás buenos resultados. Regresa al inicio y mira el video explicativo de como cargar los rostros y obtener los diseños"
    
    
    
    
    
    
    
    
    
    
    
@app.route('/ver_ultimo_resultado', methods=['GET'])
def ver_ultimo_resultado():
    static_folder = os.path.join(os.getcwd(), 'static')
    files = os.listdir(static_folder)

    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        return jsonify({'status': 'error', 'message': 'No hay imágenes disponibles'})

    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(static_folder, x)), reverse=True)

    last_image_path = os.path.join('static', image_files[0])
    print(f'Ruta de la última imagen: {last_image_path}')  # Agrega esta línea para imprimir la ruta en la consola de Flask

    return jsonify({'status': 'success', 'image_path': last_image_path})
    

@app.route('/index', methods=['GET', 'POST'])
def index():
    global a
    result_image_path = None
    
     # Debes obtener el valor real de 'a' según tus necesidades

    result_image_path = None  # Inicializa la variable result_image_path fuera del bloque condicional

    if request.method == 'POST':
        # Lógica para manejar solicitudes POST
        imagefilename = request.form.get('imagefilename', '')
        print("Nombre de la imagen de fondo recibido en Flask:", imagefilename)
        imfondo_path = session ["imafondo_path"]
        imfondo_path = construir_imfondo(imagefilename)
        
        static_dir = os.path.join('static')
        image_files = [f for f in os.listdir(static_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        if image_files:
            last_image = max(image_files, key=lambda x: os.path.getctime(os.path.join(static_dir, x)))
            result_image_path = os.path.join('static', last_image)
        
            print("Result Image Path:", result_image_path)

        
        else:
            result_image_path = None
        imagen_final
        result_image=result_image_path
        session["result_image"] = result_image
        return render_template('index.html' ,a=a)

    # Lógica para manejar solicitudes GET
    imagefilename = str(request.args.get('imagefilename', ''))
    imfondo_path = construir_imfondo(imagefilename)
    return render_template('index.html', imfondo=imfondo_path)



@app.route('/seleccion_estilo')
def seleccion_estilo():
    return render_template('seleccion_estilo.html')






@app.route('/disenos_una_persona')
def disenos_una_persona():
    
    b = "individuales"
    session["b"] = b
    # Ruta a la carpeta de imágenes para 1 persona
    folder_path = os.path.join('static', "individuales", 'reducido')
    # Obtener la lista de nombres de archivos en la carpeta
    image_files = os.listdir(folder_path)
    # Construir la ruta completa para cada imagen
    image_paths = [filename for filename in image_files]
    imagefilename = request.args.get('imagefilename', '')
   
    imfondo_path = os.path.join('uploads', "imgfondo")
    return render_template('galeria.html', image_paths=image_paths, imfondo=imfondo_path)

@app.route('/disenos2personas')
def disenos_una_persona2():
    
    
    b = "dobles"
    session["b"] = b
    # Ruta a la carpeta de imágenes para 1 persona
    folder_path = os.path.join('static', "dobles", 'reducido')
    # Obtener la lista de nombres de archivos en la carpeta
    image_files = os.listdir(folder_path)
    # Construir la ruta completa para cada imagen
    image_paths = [filename for filename in image_files]
    imagefilename = request.args.get('imagefilename', '')
    imfondo_path = construir_imfondo(imagefilename)
    return render_template('galeria2.html', image_paths=image_paths, imfondo=imfondo_path)

@app.route('/disenos3personas')
def disenos_una_persona3():
    
  
    b = "triples"
    session["b"] = b
    # Ruta a la carpeta de imágenes para 1 persona
    folder_path = os.path.join('static', "triples", 'reducido')
    # Obtener la lista de nombres de archivos en la carpeta
    image_files = os.listdir(folder_path)
    # Construir la ruta completa para cada imagen
    image_paths = [filename for filename in image_files]
    imagefilename = request.args.get('imagefilename', '')
    imfondo_path = construir_imfondo(imagefilename)
    return render_template('galeria3.html', image_paths=image_paths, imfondo=imfondo_path)



@app.route('/disenos4personas')
def disenos_una_persona4():
    
   
    b = "cuadruples"
    session["b"] = b
    # Ruta a la carpeta de imágenes para 1 persona
    folder_path = os.path.join('static', "cuadruples", 'reducido')
    # Obtener la lista de nombres de archivos en la carpeta
    image_files = os.listdir(folder_path)
    # Construir la ruta completa para cada imagen
    image_paths = [filename for filename in image_files]
    imagefilename = request.args.get('imagefilename', '')
    imfondo_path = construir_imfondo(imagefilename)
    return render_template('galeria4.html', image_paths=image_paths, imfondo=imfondo_path)

@app.route('/disenos5personas')
def disenos_una_persona5():
    
   
    b = "quintuples"
    session["b"] = b
    # Ruta a la carpeta de imágenes para 1 persona
    folder_path = os.path.join('static', "quintuples", 'reducido')
    # Obtener la lista de nombres de archivos en la carpeta
    image_files = os.listdir(folder_path)
    # Construir la ruta completa para cada imagen
    image_paths = [filename for filename in image_files]
    imagefilename = request.args.get('imagefilename', '')
    imfondo_path = construir_imfondo(imagefilename)
    return render_template('galeria5.html', image_paths=image_paths, imfondo=imfondo_path)

@app.route('/upload_fondo', methods=['POST'])
def upload_fondo():
    codigouser = session.get('codigouser')

    if not codigouser:
        return jsonify({'status': 'error', 'message': 'El código de usuario no está disponible en la sesión.'})

    try:
        fondo_file = request.files['file']
        if fondo_file:
            # Verifica si la carpeta codigouser existe, y créala si no existe
            user_folder = os.path.join('uploads', codigouser)
            if not os.path.exists(user_folder):
                os.makedirs(user_folder)

            fondo_file.save(os.path.join(user_folder, 'imgfondo'))
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'No se seleccionó ninguna imagen de fondo.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/procesar', methods=['POST'])
def procesar():
   
    
    codigouser = session['codigouser']
    
    rename_images()
    data = request.get_json()
    imagefilename_from_form = data.get('imagefilename', '')
    
    imfondo_path = construir_imfondo(imagefilename_from_form)
    carpeta_destino = os.path.join("uploads", codigouser)
    imfondo_path = session["imfondo_path"]
    codigouser = session ["codigouser"]
    imfondo_path = os.path.join('uploads', codigouser, "imgfondo")
    print("Ruta de la imagen de fondo:", imfondo_path)
    

    
    
    
    
    img = cv2.imread(imfondo_path)

    if img is None:
        print("Error al cargar la imagen.")
        return jsonify({'status': 'error', 'message': 'Error al cargar la imagen'})

    faces = app_insightface.get(img)
    user_ip = codigouser
    # Ordenar las caras por la posición del cuadro delimitador
    faces = sorted(faces, key=lambda x: x['bbox'][0])
    
    # Almacena los datos de los rostros en una lista
    folder_path = os.path.join('uploads', user_ip)
    # Obtener la lista de nombres de archivos ordenada
    file_names = sorted(os.listdir(folder_path))

    # Almacena los datos de los rostros en una lista
    faces_data = []
    faces_data.sort(key=lambda x: x['source_face']['bbox'][0])
    for i, source_face in enumerate(faces):
        bbox = source_face["bbox"]
        bbox = [int(b) for b in bbox]

        # Si hay un archivo correspondiente en la carpeta, usa su información
        if i < len(file_names):
            file_name = file_names[i]
            img_persona_path = os.path.join('uploads', user_ip, file_name)
        else:
            # En caso de que no haya suficientes archivos en la carpeta
            img_persona_path = None
        
        # Almacena los datos del rostro actual en la lista faces_data
        current_face_data = {
            'source_face': source_face,
            'img_persona_path': img_persona_path
        }
        faces_data.append(current_face_data)
    faces_data.sort(key=lambda x: os.path.basename(x['img_persona_path']))
    
    
    # Procesa los datos almacenados para generar las imágenes finales
    for i, face_data in enumerate(faces_data):
        source_face = face_data['source_face']
        img_persona_path = face_data['img_persona_path']
        
        img_persona = cv2.imread(img_persona_path)
        remp_faces = app_insightface.get(img_persona)
        remp_faces = remp_faces[0]
        img = swapper.get(img, source_face, remp_faces, paste_back=True)
        
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_number = str(random.randint(100000000, 999999999))
        
        print("Valor de user_ip:", user_ip)
        unique_name = f"output_image_{timestamp}_{random_number}_{user_ip}_{i}.jpg"
        
        
        print("Nombre de archivo único:", unique_name)
        output_path = os.path.join('static', unique_name)
        
    
        
        cv2.imwrite(output_path, img)
        print("UNIQUE NAME ES", unique_name)
        print(session)
    

    # Devuelve la última imagen generada como resultado
    unique_name = unique_name
    print ("dasdasfavaa", unique_name)
    result_image = output_path
    session['unique_name'] = unique_name
    session["result_image"] = result_image
    shutil.rmtree(os.path.join('uploads', codigouser))
    print(f"Carpeta eliminada exitosamente.")
    print(session)
    

    # Pasar la variable unique_number al template
    return render_template('imagen_final.html', result_image=unique_name, unique_name=unique_name)




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))










