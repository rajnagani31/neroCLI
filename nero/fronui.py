from textual.app import App
from textual.widgets import Header, Footer, Static

class NeroCLI(App):

    def compose(self):
        input()
        yield Header()
        yield Static("Welcome to NeroCLI 🚀")
        yield Footer()

if __name__ == "__main__":
    NeroCLI().run()