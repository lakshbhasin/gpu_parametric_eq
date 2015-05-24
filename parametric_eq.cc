#include "parametric_eq.hh"

int main(int argc, char *argv[])
{
    /* Argument parsing */

    // There should be 4 arguments.
    if (argc != 5)
    {
        usage(argv[0]);
    }

    deploy(argv[1], atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    processSound();
}
