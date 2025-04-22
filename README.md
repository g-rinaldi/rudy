# Rudy: A Rudimental Graph Generator

Rudy is a machine-independent graph generator originally written by Giovanni Rinaldi in 1995 using CWEB and components of the Stanford GraphBase (SGB) (see below). It allows for the specification of diverse graphs via command-line arguments, avoiding the need for explicit node/edge lists.

This repository contains a modernized version of Rudy. While preserving the original functionality, this version uses C++ (`rudy.cpp`) for the main application, alongside supporting C modules (`gauss.c` and components from the SGB library located in the `sgb/` directory).

## History and Availability

The original CWEB version was freely distributed via Christoph Helmberg's homepage ([rudy](https://www-user.tu-chemnitz.de/~helmberg/rudy.tar.gz)) and Yinyu Ye maintains another version within the [Gset](https://web.stanford.edu/~yyye/yyye/Gset) collection. This repository provides an updated, more readily compilable version using standard C/C++ compilers.

## Key Features

*   **Machine-Independent Generation:** Define graphs using command-line parameters.
*   **Diverse Graph Types:** Capable of producing various types of graphs (refer to program usage for details).
*   **SGB Integration:** Leverages core components of Donald Knuth's Stanford GraphBase, including its robust random number generator.

## License

*   **Rudy Code (`rudy.cpp`, `gauss.c`):** The software contained in the root directory can be freely copied, modified, and redistributed provided that credit is given to the original author, Giovanni Rinaldi.
*   **SGB Library (`sgb/` directory):** The files contained in the `sgb` directory are part of the Stanford GraphBase and are copyright © 1993 by Stanford University. They may be freely copied and distributed, provided that no changes whatsoever are made. See the file `sgb/boilerplate.w` for the full SGB copyright notice.

## Citation

If you use this software in your research, please cite it using the following BibTeX entry:

```bibtex
@misc{rudy,
  author =        {Giovanni Rinaldi},
  title =         {{Rudy: A Rudimental Graph Generator}},
  year =          {1995},
  howpublished =  {GitHub repository},
  url =           {https://github.com/g-rinaldi/rudy}
}
```

## File list

- In the root directory:
    - rudy.cpp (The main code)
    - gauss.c (A generator of the Gaussian N(0,1) random numbers)
    - Makefile (The build script)

- In the sgb directory (a subset of the Stanford GraphBase library):
    - boilerplate.w (The license notice of the SGB library)
    - gb_basic.[w,c,h]
    - gb_flip.[w,c,h]
    - gb_graph.[w,c,h]
    - gb_rand.[w,c,h]
    - gb_types.w

    The *.c and the .h files are obtained from the corresponding *.w files via the CWEB compiler `ctangle`. They are kept in the directory specifically to ensure that the code can be compiled in cases where `ctangle` is not installed.    

## Prerequisites

To build this project, you will need:

1.  A C compiler (e.g., `gcc`)
2.  A C++ compiler (e.g., `g++`)
3.  The `make` utility

The necessary C source files (`*.c`) and the main C++ source file (`rudy.cpp`) should be present in the root directory of this repository.

## Building

The build process is managed by the `Makefile` file. To compile the project:

1.  Navigate to the root directory of the repository in your terminal.
2.  Run the `make` command:
    ```bash
    make
    ```

This command will:
*   Create a `bin/` directory if it doesn't exist.
*   Compile all `.c` files using `gcc` with `-O2` optimization and warning suppression (`-w`).
*   Compile `rudy.cpp` using `g++` with `-O2` optimization.
*   Place all intermediate object files (`.o`) into the `bin/` directory.
*   Link the object files into a final executable named `rudy` located in the `bin/` directory.
*   Remove the intermediate object files from the `bin/` directory after successful linking.

The final executable will be located at `bin/rudy`.

## Running

After successful compilation, you can run the program from the root directory:

```bash
./bin/rudy [arguments]
```

## The Stanford GraphBase

“The [Stanford GraphBase](https://www-cs-faculty.stanford.edu/~knuth/sgb.html)
(SGB) is a collection of datasets and computer programs that generate and
examine a wide variety of graphs and networks.” It was developed and published
by [Donald E. Knuth](https://www-cs-faculty.stanford.edu/~knuth) in 1993. The
fully documented source code is available for download from [Stanford
University](https://ftp.cs.stanford.edu/pub/sgb/sgb.tar.gz) and in the book
“The Stanford GraphBase, A Platform for Combinatorial Computing,” published
jointly by ACM Press and Addison-Wesley Publishing Company in 1993. (This book
contains several chapters with additional information not available in the
electronic distribution.)

### Prerequisites

The source code of SGB is written in accordance with the rules of the
[Literate Programming](https://www-cs-faculty.stanford.edu/~knuth/lp.html)
paradigm, so you need to make sure that your computer supports the
[CWEB](https://www-cs-faculty.stanford.edu/~knuth/cweb.html) system. The CWEB
sources are available for download from [Stanford
University](https://ftp.cs.stanford.edu/pub/cweb/cweb.tar.gz). Bootstrapping
CWEB on Unix systems is elementary and documented in the CWEB distribution;
pre-compiled binary executables of the CWEB tools for Win32 systems are
available from
[www.literateprogramming.com](http://www.literateprogramming.com).
