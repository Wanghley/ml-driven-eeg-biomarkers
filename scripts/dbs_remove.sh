#!/bin/bash
# DBS Artifact Removal - Simple Wrapper Script
# Makes it easy to run the CLI tool without remembering full command names

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_banner() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║          DBS Artifact Removal - Quick Run Wrapper              ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_methods() {
    echo -e "${YELLOW}Available Methods:${NC}"
    echo "  1) spectrum_fit    - Fast, strong removal (RECOMMENDED)"
    echo "  2) wiener          - Best brain preservation (needs baseline)"
    echo "  3) zapline         - Multi-channel spatial filtering"
    echo "  4) hampel_freq     - Spectral peak detection"
    echo "  5) hampel_time     - Transient pulse removal (fastest)"
    echo ""
}

show_help() {
    print_banner
    echo "Usage: ./dbs_remove.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  -i, --interactive        Run interactive mode (select everything)"
    echo "  -m, --method METHOD      Select method (spectrum_fit, wiener, etc.)"
    echo "  -f, --file FILE          Input EDF file"
    echo "  -b, --baseline BASE      Baseline file (for wiener method)"
    echo "  -o, --output DIR         Output directory"
    echo "  --batch                  Process all files in data/XU/"
    echo "  -l, --list               List available methods"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Interactive mode (simplest)"
    echo "  ./dbs_remove.sh -i"
    echo ""
    echo "  # Spectrum-Fit default"
    echo "  ./dbs_remove.sh -f data/XU/XUAWAKE7_deidentified.edf"
    echo ""
    echo "  # Wiener with baseline"
    echo "  ./dbs_remove.sh -m wiener -f XUAWAKE7_deidentified.edf -b XUAWAKEPRE_deidentified.edf"
    echo ""
    echo "  # Batch process all files"
    echo "  ./dbs_remove.sh --batch"
    echo ""
}

# Default values
INTERACTIVE=false
METHOD="spectrum_fit"
INPUT_FILE=""
BASELINE_FILE=""
OUTPUT_DIR="data/processed/"
BATCH=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--interactive)
            INTERACTIVE=true
            shift
            ;;
        -m|--method)
            METHOD="$2"
            shift 2
            ;;
        -f|--file)
            INPUT_FILE="$2"
            shift 2
            ;;
        -b|--baseline)
            BASELINE_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch)
            BATCH=true
            shift
            ;;
        -l|--list)
            print_banner
            print_methods
            exit 0
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

print_banner

# Check if environment is activated
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Error: Python not found${NC}"
    echo "Make sure to activate conda environment first:"
    echo "  conda activate eeg_thesis_v1"
    exit 1
fi

# Build Python command
CMD="python remove_dbs_artifacts_cli.py"

if [ "$INTERACTIVE" = true ]; then
    echo -e "${GREEN}Starting interactive mode...${NC}"
    echo ""
    eval "$CMD"
elif [ "$BATCH" = true ]; then
    echo -e "${GREEN}Batch processing all files in data/XU/...${NC}"
    CMD="$CMD --batch --method $METHOD --output-dir $OUTPUT_DIR"
    eval "$CMD"
elif [ -n "$INPUT_FILE" ]; then
    echo -e "${GREEN}Processing: $INPUT_FILE${NC}"
    CMD="$CMD --input-file $INPUT_FILE --method $METHOD --output-dir $OUTPUT_DIR"
    
    # Add baseline if method is wiener
    if [ "$METHOD" = "wiener" ]; then
        if [ -z "$BASELINE_FILE" ]; then
            echo -e "${YELLOW}⚠ Warning: Wiener method requires baseline file${NC}"
            echo "Usage: ./dbs_remove.sh -m wiener -f FILE -b BASELINE"
            exit 1
        fi
        CMD="$CMD --baseline-file $BASELINE_FILE"
    fi
    
    eval "$CMD"
else
    echo -e "${YELLOW}No input file specified.${NC}"
    echo ""
    print_methods
    echo "Use one of these:"
    echo "  ./dbs_remove.sh -i              # Interactive mode"
    echo "  ./dbs_remove.sh --batch         # Batch all files"
    echo "  ./dbs_remove.sh -f FILE         # Process FILE"
    echo ""
    exit 0
fi
