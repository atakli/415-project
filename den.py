import PySimpleGUI as sg

layout = [[sg.Text('Enter a Number')],
          [sg.Input()],
          [sg.OK()] ]

event, values = sg.Window('Enter a number example', layout).Read()

# breakpoint()
sg.Popup(event, values['0'])