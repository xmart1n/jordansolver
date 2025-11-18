from flask import Flask, jsonify, render_template, make_response, request
import backend.solver as solver
import backend.solver_auxilary as aux

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("index.html")


@app.post("/calculate")
def calc():
    matrix = request.json["matrix"]
    inst = solver.Solver()
    inst.process(matrix)
    return inst.get_datapack()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
