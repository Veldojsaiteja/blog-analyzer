from flask import Flask, render_template, request

from main import analyze_blog_content

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        blog_content = request.form['blog_content']
        analysis_results = analyze_blog_content(blog_content)
        return render_template('result.html', analysis_results=analysis_results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, host = '0.0.0.0')
