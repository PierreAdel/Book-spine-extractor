import PySimpleGUI as sg
import sys

if len(sys.argv) == 1:
    event, values = sg.Window('cvc app',
                    [[sg.Text('Image to open')],
                    [sg.In(), sg.FileBrowse()],
                    [sg.Open(), sg.Cancel()]]).read(close=True)
    fname = values[0]
else:
    fname = sys.argv[1]

if not fname:
    sg.popup("Cancel", "No filename supplied")
    raise SystemExit("Cancelling: no filename supplied")
else:
    sg.popup('The filename you chose was', fname)
    # then
    # call the script
    # display the result
