NetTracer3DV2 is an improvement on V1 for general use purposes.
It features easier useability, more powerful and faster algorithms, and more
network-related analytical tools (many child funcitons of which are not used in this experiment specfically)

Although a lot of the data in the preprint was computed on V1 (which yields largely similar results in the use cases presented), I would recommend trying out many of V2's functions
for validation of V1 outputs. In the future, V2 will be the primary version utilized.

While V1 and related are designed to be run from the command line, then prompt the user for
related inputs, then execute and produce some result, V2 is structured around initializing
and performing methods/functions on a Network_3D Class object. V2 therefore is meant to be
installed as a package and executed from external notebooks for much greater flexiblility
in usage.

Note that V2 is currently available on PyPi and installable with pip. Likewise it can be installed with 'pip install nettracer3d' onto python3 compatible machines. There are other installation options for GPU integration. NetTracer3D is an ongoing project and is liable to be updated in the future. Please see its pyproject page for the latest builds: https://pypi.org/project/nettracer3d/

V2 has a user manual that is included in this folder and includes more documentation.