Procesamiento de Lenguaje Natural - UBA 2018
============================================


Instalación
-----------

1. Se necesita el siguiente software:

   - Git
   - Pip
   - Python 3.4 o posterior
   - TkInter
   - Virtualenv

   En un sistema basado en Debian (como Ubuntu), se puede hacer::

    sudo apt-get install git python-pip python3 python3-tk virtualenv

2. Crear y activar un nuevo
   `virtualenv <http://virtualenv.readthedocs.org/en/latest/virtualenv.html>`_.
   Recomiendo usar `virtualenvwrapper
   <http://virtualenvwrapper.readthedocs.org/en/latest/install.html#basic-installation>`_.
   Se puede instalar así::

    sudo pip install virtualenvwrapper

   Y luego agregando la siguiente línea al final del archivo ``.bashrc``::

    [[ -s "/usr/local/bin/virtualenvwrapper.sh" ]] && source "/usr/local/bin/virtualenvwrapper.sh"

   Para crear y activar nuestro virtualenv::

    mkvirtualenv --python=/usr/bin/python3 pln

3. Bajar el código::

    git clone https://github.com/PLN-FaMAF/PLN-UBA2018.git

4. Instalarlo::

    cd PLN-UBA2018
    pip install -r requirements.txt


Ejecución
---------

1. Activar el entorno virtual con::

    workon pln

2. Correr el script que uno quiera. Por ejemplo::

    python languagemodeling/scripts/train.py -h


Testing
-------

Correr nose::

    nosetests


Chequear Estilo de Código
-------------------------

Correr flake8 sobre el paquete o módulo que se desea chequear. Por ejemplo::

    flake8 languagemodeling
