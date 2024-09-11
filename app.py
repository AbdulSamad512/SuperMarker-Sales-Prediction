import pandas as pd 
import numpy as np 
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)
data = pd.read_csv('rf2.csv')
print(data.head())  # Debugging: Print the first few rows of the DataFrame
pipe = joblib.load(r'C:\Users\AM BUSINESS\Desktop\Model Deployment\pipe_rf_item.sav')


@app.route('/')
def index():
    stores = sorted(data['Store'].unique())
    products = sorted(data['Product Description'].unique())
    groups = sorted(data['Group'].unique())
    categorys = sorted(data['Category'].unique())
    print(stores)  # Debugging: Print the list of stores
    print(products)  # Debugging: Print the list of products
    print(groups)
    print(categorys)
    return render_template('index.html',stores=stores, products=products, groups=groups, categorys=categorys)

@app.route('/predict',methods=['POST'])
def predict():
    day = request.form.get('Day')  # Column 'Day' ke sath case match karein
    month = request.form.get('Month')  # Column 'Month' ke sath case match karein
    year = request.form.get('Year')  # Column 'Year' ke sath case match karein
    itemcode = request.form.get('ItemCode')  # Column 'ItemCode' ke sath match karein
    store = request.form.get('Store')  # Column 'Store' ke sath match karein
    product = request.form.get('Product Description')  # Column 'Product Description' ka exact naam match karein
    group = request.form.get('Group')  # Column 'Group' ke sath case match karein
    category = request.form.get('Category')  # Column 'Category' ke sath case match karein
    cost = request.form.get('total cost')  # Column 'total cost' ka exact naam match karein

    print(day,month,year,itemcode,store,product,group,category,cost)
    input = pd.DataFrame([[day,month,year,itemcode,store,product,group,category,cost]],columns=['Day','Month','Year','ItemCode','Store','Product Description','Group','Category','total cost'])
    prediction = np.exp(pipe.predict(input)[0])


    return str(np.round(prediction,2))

if __name__ == "__main__":
    app.run(debug=True, port=5002)
