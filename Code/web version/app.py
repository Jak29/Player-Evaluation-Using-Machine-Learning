from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import re

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Soccer Predictions</title>
    <style>
        body {
            display: grid;
            font-family: Arial, sans-serif;
            margin: 20px;
            padding: 20px;
        }
        h2 {
            margin-top: 40px;
            margin-bottom: 0px;
        }
        .container {
            margin-bottom: 20px;
        }
        label {
            margin-right: 15px;
        }
        button {
            margin-top: 20px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
        }
        #deselectAllAttributes {
            background-color: #ff0000;
            color: #ffffff;
        }   
        #deselectAllLeagues {
            background-color: #ff0000;
            color: #ffffff;
        }   
        #predictions {
            margin-top: 20px;
            padding: 10px;
            background-color: #f2f2f2;
            border: dashed 1px gray;
        }
        .divisions-container {
            display: grid;
            grid-template-columns: repeat(6, 150px);
            column-gap: 20px;
            row-gap: 10px;
            max-width: 100%;
            margin: auto;
        }
        .divisions {
            display: flex;
            align-items: center;
        }
        .divisions label {
            margin-left: 5px;
            white-space: nowrap;
        }
        .attributes-container {
            display: grid;
            grid-template-columns: repeat(6, 150px);
            column-gap: 20px;
            row-gap: 10px;
            max-width: 100%;
            margin: auto;
        }
        .attribute {
            display: flex;
            align-items: center;
        }
        .attribute label {
            margin-left: 5px;
            white-space: nowrap;
        }
    </style>
</head>
<body>
    <h1>Select Divisions and Player Attributes</h1>
    <form id="divisionForm">
        <h2>Divisions</h2>

        <div class="container">
            <button type="button" id="selectAllLeagues">Select All Leagues</button>
            <button type="button" id="deselectAllLeagues">Deselect All Leagues</button>
        </div>

        <div class="divisions-container">
        
            <label for="English">Premier League</label>
            <input type="checkbox" id="English" name="divisions" value="English Premier Division" class="league-checkbox">

            <label for="Ligue1">Ligue 1</label>
            <input type="checkbox" id="Ligue1" name="divisions" value="Ligue 1 Conforama" class="league-checkbox">

            <label for="Bundesliga">Bundesliga</label>
            <input type="checkbox" id="Bundesliga" name="divisions" value="Bundesliga" class="league-checkbox">

            <label for="Spanish">La Liga</label>
            <input type="checkbox" id="Spanish" name="divisions" value="Spanish First Division" class="league-checkbox">

            <label for="SerieA">Serie A</label>
            <input type="checkbox" id="SerieA" name="divisions" value="Italian Serie A" class="league-checkbox">

            <label for="Portuguese">Primeira Liga</label>
            <input type="checkbox" id="Portuguese" name="divisions" value="Portuguese Premier League" class="league-checkbox">

            <label for="Eredivisie">Eredivisie</label>
            <input type="checkbox" id="Eredivisie" name="divisions" value="Eredivisie" class="league-checkbox">

        </div>

        <h2>Player Attributes</h2>

        <div class="container">
            <button type="button" id="selectAllAttributes">Select All Attributes</button>
            <button type="button" id="deselectAllAttributes">Deselect All Attributes</button>
        </div>

        <div class="attributes-container">

                <label for="Age">Age</label>
                <input type="checkbox" id="Age" name="attributes" value="Age" class="attribute-checkbox">

                <label for="Position">Position</label>
                <input type="checkbox" id="Position" name="attributes" value="Position" class="attribute-checkbox">
                
                <label for="TrueHeight">TrueHeight</label>
                <input type="checkbox" id="TrueHeight" name="attributes" value="TrueHeight" class="attribute-checkbox">
                
                <label for="Acc">Acceleration</label>
                <input type="checkbox" id="Acc" name="attributes" value="Acc" class="attribute-checkbox">
                
                <label for="Wor">Work Rate</label>
                <input type="checkbox" id="Wor" name="attributes" value="Wor" class="attribute-checkbox">
                
                <label for="Vis">Vision</label>
                <input type="checkbox" id="Vis" name="attributes" value="Vis" class="attribute-checkbox">
                
                <label for="Thr">Throwing</label>
                <input type="checkbox" id="Thr" name="attributes" value="Thr" class="attribute-checkbox">
                
                <label for="Tec">Technique</label>
                <input type="checkbox" id="Tec" name="attributes" value="Tec" class="attribute-checkbox">

                <label for="Tea">Teamwork</label>
                <input type="checkbox" id="Tea" name="attributes" value="Tea" class="attribute-checkbox">
                
                <label for="Tck">Tackling</label>
                <input type="checkbox" id="Tck" name="attributes" value="Tck" class="attribute-checkbox">
                
                <label for="Str">Strength</label>
                <input type="checkbox" id="Str" name="attributes" value="Str" class="attribute-checkbox">
                
                <label for="Sta">Stamina</label>
                <input type="checkbox" id="Sta" name="attributes" value="Sta" class="attribute-checkbox">
                
                <label for="TRO">Throw Ons</label>
                <input type="checkbox" id="TRO" name="attributes" value="TRO" class="attribute-checkbox">
                
                <label for="Ref">Reflexes</label>
                <input type="checkbox" id="Ref" name="attributes" value="Ref" class="attribute-checkbox">
                
                <label for="Pun">Punching</label>
                <input type="checkbox" id="Pun" name="attributes" value="Pun" class="attribute-checkbox">
                
                <label for="Pos">Positioning</label>
                <input type="checkbox" id="Pos" name="attributes" value="Pos" class="attribute-checkbox">
                
                <label for="Pen">Penalties</label>
                <input type="checkbox" id="Pen" name="attributes" value="Pen" class="attribute-checkbox">
                
                <label for="Pas">Passing</label>
                <input type="checkbox" id="Pas" name="attributes" value="Pas" class="attribute-checkbox">
                
                <label for="Pac">Pace</label>
                <input type="checkbox" id="Pac" name="attributes" value="Pac" class="attribute-checkbox">
                
                <label for="1v1">One vs One</label>
                <input type="checkbox" id="1v1" name="attributes" value="1v1" class="attribute-checkbox">
                
                <label for="OtB">Off the Ball</label>
                <input type="checkbox" id="OtB" name="attributes" value="OtB" class="attribute-checkbox">
                
                <label for="Mar">Marking</label>
                <input type="checkbox" id="Mar" name="attributes" value="Mar" class="attribute-checkbox">
                
                <label for="Lon">Long Shots</label>
                <input type="checkbox" id="Lon" name="attributes" value="Lon" class="attribute-checkbox">
                
                <label for="Ldr">Leadership</label>
                <input type="checkbox" id="Ldr" name="attributes" value="Ldr" class="attribute-checkbox">
                
                <label for="Kic">Kicking</label>
                <input type="checkbox" id="Kic" name="attributes" value="Kic" class="attribute-checkbox">
                
                <label for="Jum">Jumping</label>
                <input type="checkbox" id="Jum" name="attributes" value="Jum" class="attribute-checkbox">
                
                <label for="Hea">Heading</label>
                <input type="checkbox" id="Hea" name="attributes" value="Hea" class="attribute-checkbox">
                
                <label for="Han">Handling</label>
                <input type="checkbox" id="Han" name="attributes" value="Han" class="attribute-checkbox">
                
                <label for="Fre">Free Kick</label>
                <input type="checkbox" id="Fre" name="attributes" value="Fre" class="attribute-checkbox">
                
                <label for="Fir">First Touch</label>
                <input type="checkbox" id="Fir" name="attributes" value="Fir" class="attribute-checkbox">
                
                <label for="Fin">Finishing</label>
                <input type="checkbox" id="Fin" name="attributes" value="Fin" class="attribute-checkbox">
                
                <label for="Ecc">Eccentricity</label>
                <input type="checkbox" id="Ecc" name="attributes" value="Ecc" class="attribute-checkbox">
                
                <label for="Dri">Dribbling</label>
                <input type="checkbox" id="Dri" name="attributes" value="Dri" class="attribute-checkbox">
                
                <label for="Det">Determination</label>
                <input type="checkbox" id="Det" name="attributes" value="Det" class="attribute-checkbox">
                
                <label for="Dec">Decision Making</label>
                <input type="checkbox" id="Dec" name="attributes" value="Dec" class="attribute-checkbox">
                
                <label for="Cro">Crossing</label>
                <input type="checkbox" id="Cro" name="attributes" value="Cro" class="attribute-checkbox">
                
                <label for="Cor">Corners</label>
                <input type="checkbox" id="Cor" name="attributes" value="Cor" class="attribute-checkbox">
                
                <label for="Cnt">Control</label>
                <input type="checkbox" id="Cnt" name="attributes" value="Cnt" class="attribute-checkbox">
                
                <label for="Cmp">Composure</label>
                <input type="checkbox" id="Cmp" name="attributes" value="Cmp" class="attribute-checkbox">
                
                <label for="Com">Communication</label>
                <input type="checkbox" id="Com" name="attributes" value="Com" class="attribute-checkbox">
                
                <label for="Cmd">Command of Area</label>
                <input type="checkbox" id="Cmd" name="attributes" value="Cmd" class="attribute-checkbox">
                
                <label for="Bal">Balance</label>
                <input type="checkbox" id="Bal" name="attributes" value="Bal" class="attribute-checkbox">
                
                <label for="Ant">Anticipation</label>
                <input type="checkbox" id="Ant" name="attributes" value="Ant" class="attribute-checkbox">
                
                <label for="Agi">Agility</label>
                <input type="checkbox" id="Agi" name="attributes" value="Agi" class="attribute-checkbox">
                
                <label for="Agg">Aggression</label>
                <input type="checkbox" id="Agg" name="attributes" value="Agg" class="attribute-checkbox">
        
                <label for="Aer">Aerial Reach</label>
                <input type="checkbox" id="Aer" name="attributes" value="Aer" class="attribute-checkbox">
        </div>
        <button type="submit">Get Predictions</button>
    </form>
    <h2>Predictions and Feature Importances:</h2>
    <div id="predictions"></div>


    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <script>
        $(document).ready(function() {
            $('#divisionForm').submit(function(e) {
                e.preventDefault();
                var selectedDivisions = $('input[name="divisions"]:checked').map(function() {
                    return this.value;
                }).get();
                var selectedAttributes = $('input[name="attributes"]:checked').map(function() {
                    return this.value;
                }).get();
                $.ajax({
                    type: "POST",
                    url: "/get_predictions",
                    contentType: "application/json",
                    data: JSON.stringify({divisions: selectedDivisions, attributes: selectedAttributes}),
                    success: function(response) {
                        var results = "";
                        results += "<p>Mean RMSE: " + response.mean_rmse.toFixed(2) + "</p>";
                        results += "<p>Mean MAE: " + response.mean_mae.toFixed(2) + "</p>";
                        results += "<p>Mean R-squared: " + response.mean_r_squared.toFixed(4) + "</p>";
                        results += "<h3>Feature Importances:</h3>";
                        results += "<ul>";
                        $.each(response.feature_importances, function(feature, importance) {
                            results += "<li>" + feature + ": " + importance.toFixed(4) + "</li>";
                        });
                        results += "</ul>";
                        $('#predictions').html(results);
                    }
                });
            });
            

            $('#selectAllAttributes').click(function() {
                $('.attribute-checkbox').prop('checked', true);
            });

            $('#deselectAllAttributes').click(function() {
                $('.attribute-checkbox').prop('checked', false);
            });

            $('#selectAllLeagues').click(function() {
                $('.league-checkbox').prop('checked', true);
            });

            $('#deselectAllLeagues').click(function() {
                $('.league-checkbox').prop('checked', false);
            });

        });
    </script>
</body>
</html>
"""

def assign_position_value(position):
    position_values = {
        'ST': 15,
        'M': 16,
        'D': 19,
        'GK': 17,
    }
    found_positions = re.findall(r'\b(ST|M|D|GK)\b', position)
    max_value = 0
    for pos in found_positions:
        if pos in position_values:
            max_value = max(max_value, position_values[pos])
    return max_value

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/get_predictions', methods=['POST'])
def get_predictions():
    data = request.get_json()
    divisions = data['divisions']
    attributes = data['attributes'] + ['TrueValue']

    df = pd.read_csv("../2020.csv")
    df = df[df["Division"].isin(divisions)]
    df = df[attributes]

    df['PositionValue'] = df['Position'].apply(assign_position_value)
    df = df.drop("Position", axis=1)

    X = df.drop("TrueValue", axis=1)
    y = df["TrueValue"]

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    mae_scores = []
    r_squared_scores = []
    feature_importances = np.zeros(X.shape[1])

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.15, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r_squared = r2_score(y_test, predictions)

        feature_importances += model.feature_importances_

        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r_squared_scores.append(r_squared)

    feature_importances /= kf.get_n_splits()

    feature_importance_dict = dict(zip(X.columns, feature_importances))

    results = {
        'mean_rmse': np.mean(rmse_scores),
        'mean_mae': np.mean(mae_scores),
        'mean_r_squared': np.mean(r_squared_scores),
        'feature_importances': feature_importance_dict
    }

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
