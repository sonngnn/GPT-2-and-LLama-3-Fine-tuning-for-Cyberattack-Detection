#!/bin/bash

# Multi-Feature LLaMA Training Pipeline avec Feature Engineering
# Ce script utilise le nouveau orchestrateur pour traiter plusieurs configurations en parallèle

# Chargement des modules
echo "Chargement des modules..."
module load gcc
module load arrow/21.0.0
module load cuda/12.6

echo "=== Multi-Feature LLaMA Pipeline avec Feature Engineering ==="

# Configuration par défaut
DEFAULT_FEATURES="8 12 16"
DEFAULT_CSV="./data/CICIoT2023_attacks_benign_CTGAN_V2.csv"
DEFAULT_MAX_CONCURRENT=3

# Parsing des arguments
FEATURES_LIST="$DEFAULT_FEATURES"
CSV_PATH="$DEFAULT_CSV"
MAX_CONCURRENT="$DEFAULT_MAX_CONCURRENT"
DRY_RUN=""
PREPROCESSING_ONLY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --features)
            shift
            FEATURES_LIST=""
            # Collecter tous les nombres jusqu'au prochain argument
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                FEATURES_LIST="$FEATURES_LIST $1"
                shift
            done
            ;;
        --csv)
            CSV_PATH="$2"
            shift 2
            ;;
        --max_concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN="--dry_run"
            shift
            ;;
        --preprocessing_only)
            PREPROCESSING_ONLY="--preprocessing_only"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "OPTIONS:"
            echo "  --features N1 N2 N3...    Nombres de features à traiter (défaut: $DEFAULT_FEATURES)"
            echo "  --csv PATH               Chemin vers le CSV (défaut: $DEFAULT_CSV)"
            echo "  --max_concurrent N       Max entraînements simultanés (défaut: $DEFAULT_MAX_CONCURRENT)"
            echo "  --dry_run               Mode simulation"
            echo "  --preprocessing_only    Preprocessing seulement"
            echo "  --help                  Aide"
            echo ""
            echo "EXEMPLES:"
            echo "  $0                                    # Configuration par défaut"
            echo "  $0 --features 8 12 16 20             # Features spécifiques"
            echo "  $0 --preprocessing_only              # Preprocessing seulement"
            echo "  $0 --dry_run                         # Test de configuration"
            exit 0
            ;;
        *)
            echo "Argument inconnu: $1"
            echo "Utilisez --help pour l'aide"
            exit 1
            ;;
    esac
done

# Vérification des prérequis
echo "Vérification de l'environnement..."

# CUDA
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')" || {
    echo "Erreur: Impossible de vérifier CUDA"
    exit 1
}

# GPU count
python -c "import torch; print(f'Nombre de GPUs: {torch.cuda.device_count()}')" 2>/dev/null || {
    echo "Attention: Impossible de détecter les GPUs"
}

# Variables d'environnement pour optimiser les performances
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

# Vérification du fichier CSV
if [ ! -f "$CSV_PATH" ]; then
    echo "Erreur: Fichier CSV non trouvé: $CSV_PATH"
    echo "Veuillez:"
    echo "  1. Vérifier le chemin du fichier"
    echo "  2. Ou utiliser --csv /chemin/vers/votre/fichier.csv"
    exit 1
fi

# Vérification du script d'orchestration
if [ ! -f "orchestrate_pipeline.py" ]; then
    echo "Erreur: orchestrate_pipeline.py non trouvé"
    echo "Assurez-vous que tous les scripts sont dans le répertoire courant"
    exit 1
fi

# Vérification des dépendances principales
echo "Vérification des dépendances Python..."
python -c "
import sys
required_modules = ['torch', 'transformers', 'peft', 'datasets', 'sklearn', 'pandas', 'numpy']
missing = []

for module in required_modules:
    try:
        __import__(module)
    except ImportError:
        missing.append(module)

if missing:
    print(f'Modules manquants: {missing}')
    print('Installez avec: pip install -r requirements.txt')
    sys.exit(1)
else:
    print('Toutes les dépendances sont présentes')
" || exit 1

# Création des répertoires nécessaires
echo "Création des répertoires..."
mkdir -p data outputs logs

# Affichage de la configuration
echo ""
echo "=== DÉMARRAGE DU PIPELINE ==="

# Commande d'exécution du pipeline
PIPELINE_CMD="python orchestrate_pipeline.py --features_list $FEATURES_LIST --csv_path \"$CSV_PATH\" --max_concurrent_training $MAX_CONCURRENT $DRY_RUN $PREPROCESSING_ONLY"

echo "Exécution: $PIPELINE_CMD"
echo ""

# Enregistrement du temps de début
START_TIME=$(date)
echo "Démarrage: $START_TIME"

# Exécution du pipeline
eval $PIPELINE_CMD

# Récupération du code de sortie
EXIT_CODE=$?

# Enregistrement du temps de fin
END_TIME=$(date)
echo ""
echo "=== RÉSULTAT DU PIPELINE ==="
echo "Fin: $END_TIME"

if [ $EXIT_CODE -eq 0 ]; then
    echo "STATUT: SUCCÈS"
    echo ""
    
    # Affichage des résultats si pas en dry run
    if [ -z "$DRY_RUN" ]; then
        echo "=== RÉSULTATS DISPONIBLES ==="
        
        # Vérification des datasets générés
        echo "Datasets générés:"
        for n_features in $FEATURES_LIST; do
            data_dir="data/data_${n_features}_features"
            if [ -d "$data_dir" ]; then
                dataset_size=$(du -sh "$data_dir" 2>/dev/null | cut -f1)
                echo "  ✓ $n_features features: $data_dir ($dataset_size)"
            else
                echo "  ✗ $n_features features: non généré"
            fi
        done
        
        echo ""
        
        # Vérification des modèles entraînés (si pas preprocessing only)
        if [ -z "$PREPROCESSING_ONLY" ]; then
            echo "Modèles entraînés:"
            for n_features in $FEATURES_LIST; do
                model_dir="outputs/model_${n_features}_features"
                if [ -d "$model_dir" ]; then
                    model_size=$(du -sh "$model_dir" 2>/dev/null | cut -f1)
                    echo "  ✓ $n_features features: $model_dir ($model_size)"
                else
                    echo "  ✗ $n_features features: non entraîné"
                fi
            done
        fi
        
        echo ""
        echo "Pour utiliser un modèle:"
        echo "  python inference.py  # (modifiez le chemin vers le modèle souhaité)"
        echo ""
        echo "Pour évaluer un modèle:"
        echo "  python evaluate_model.py  # (modifiez le chemin vers le modèle souhaité)"
    fi
    
else
    echo "STATUT: ÉCHEC (code $EXIT_CODE)"
    echo ""
    echo "Vérifiez les logs dans le répertoire logs/ pour plus de détails"
    
    # Affichage des dernières lignes des logs d'erreur
    echo ""
    echo "=== LOGS D'ERREUR RÉCENTS ==="
    for n_features in $FEATURES_LIST; do
        log_dir="logs/${n_features}_features"
        if [ -d "$log_dir" ]; then
            echo "--- Logs pour $n_features features ---"
            for log_file in "$log_dir"/*.log; do
                if [ -f "$log_file" ]; then
                    echo "Fichier: $(basename "$log_file")"
                    tail -n 5 "$log_file" 2>/dev/null || echo "  (vide ou inaccessible)"
                    echo ""
                fi
            done
        fi
    done
fi

echo "=== FIN DU PIPELINE ==="
exit $EXIT_CODE=== CONFIGURATION DU PIPELINE ==="
echo "Features à traiter: $FEATURES_LIST"
echo "Fichier CSV: $CSV_PATH"
echo "Max entraînements simultanés: $MAX_CONCURRENT"
echo "Mode dry run: $([ -n "$DRY_RUN" ] && echo "OUI" || echo "NON")"
echo "Preprocessing seulement: $([ -n "$PREPROCESSING_ONLY" ] && echo "OUI" || echo "NON")"
echo "================================="

# Confirmation utilisateur (sauf en dry run)
if [ -z "$DRY_RUN" ]; then
    echo ""
    echo -n "Continuer avec cette configuration ? [O/n]: "
    read -r response
    if [[ $response =~ ^[Nn] ]]; then
        echo "Pipeline annulé par l'utilisateur"
        exit 0
    fi
fi

echo ""
echo "