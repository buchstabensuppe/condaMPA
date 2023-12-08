import sysidentpy

# Laden der simulierten Daten
data = np.loadtxt("data.csv")

# Identifizierung des NARMAX-Modells
model = sysidentpy.narmax(data, 2, 2, 2, 2, 2, "linear")

# Anzeigen der Parameter des NARMAX-Modells
print(model.parameters)

# Bewertung des NARMAX-Modells
target = np.loadtxt("target.csv")
error = model.evaluate(data, target)

# Ausgabe des Fehlers des NARMAX-Modells
print(error)