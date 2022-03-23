from flask import Flask, render_template,request,redirect, send_from_directory,url_for

#Object detection
from image_detection import detect_ships, ship_detection_init

app = Flask(__name__)  

app.config['IMAGE_UPLOAD_FOLDER'] = "./static/image_uploads/"
app.config['model'], app.config['category_index'] = ship_detection_init()

@app.route('/')  
def base():  
    return render_template("base.html")  


@app.route('/image/upload')  
def image_upload():  
    return render_template("image_upload.html")
    
@app.route('/image/success', methods = ['POST'])  
def image_success():  
    if request.method == 'POST':  
        f = request.files['file']
        f.save(f"./static/image_uploads/{f.filename}")
        
        
        #formatting filename of output
        img_name, ext= f.filename.split(".")
        input_filename = f.filename
        out_filename = f"{img_name}_out.{ext}"

        #detect image
        detect_ships(f"./static/image_uploads/{input_filename}", out_filename, app.config['model'], app.config['category_index'])
        return render_template("image_success.html", input_filename=input_filename, out_filename=out_filename)
    
@app.route('/image/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='image_uploads/' + filename), code=301)
  

@app.route('/video/upload')  
def video_upload():  
    return render_template("video_upload.html")
    
@app.route('/video/success', methods = ['POST'])  
def video_success():  
    if request.method == 'POST':  
        f = request.files['file']
        f.save(f"./static/video_uploads/{f.filename}")
        
        
        #formatting filename of output
        img_name, ext= f.filename.split(".")
        input_filename = f.filename
        out_filename = f"{img_name}_out.{ext}"

        #detect image
        detect_ships(f"./static/video_uploads/{input_filename}", out_filename, app.config['model'], app.config['category_index'], False)
        return render_template("video_success.html", input_filename=input_filename, out_filename=out_filename)
    
@app.route('/video/display/<filename>')
def display_video(filename):
	#print('display_image filename: ' + filename)
	return send_from_directory(url_for('static', filename='video_uploads/' + filename), code=301, conditional=True)



if __name__ == '__main__': 
    app.run(debug = True)  