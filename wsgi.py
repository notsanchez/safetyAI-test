from app import app
import threading
from app import funcao_segundo_plano

if __name__ == "__main__":
    thread = threading.Thread(target=funcao_segundo_plano)
    thread.start()
    app.run()

