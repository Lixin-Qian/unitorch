
## Overview

> This is the structure of this library.

![Overview](/sources/images/overview.png)

#### Base Modules
Base modules are mainly designed to finish basic functions and they only focus on the implemented functionality itself.


#### Adapter Modules
Adapter modules are the adapters for base modules, to make them support different workflows. A base module may have several adapters to meet the different requirements from different workflows.


#### Command Line Interface
Command line defines a running workflow, and it will call the needed adapter modules according to the pipeline design.

