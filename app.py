from flask import Flask, request, render_template_string
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

app = Flask(__name__)

# HTML template đơn giản
HTML_PAGE = """
<!doctype html>
<title>Anomaly Detection Demo</title>
<h2>Nhập dữ liệu (cách nhau bằng dấu phẩy):</h2>
<form method=post>
  <input type=text name=data style="width:300px">
  <input type=submit value=Check>
</form>
{% if result is not none %}
  <h3>Kết quả: {{ result }}</h3>
{% endif %}
"""

# Model anomaly detection
model = IsolationForest(contamination=0.2, random_state=42)
# Train thử với dữ liệu random
train_data = np.random.randn(100, 1)
model.fit(train_data)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            values = [float(x) for x in request.form["data"].split(",")]
            values = np.array(values).reshape(-1, 1)
            prediction = model.predict(values)
            result = ["Bình thường" if p == 1 else "Bất thường" for p in prediction]
        except:
            result = "Lỗi dữ liệu nhập"
    return render_template_string(HTML_PAGE, result=result)

if __name__ == "__main__":
    app.run(debug=True)
