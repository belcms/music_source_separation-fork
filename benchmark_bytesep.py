import os
import time
import shutil
import subprocess
import pandas as pd
import museval
import psutil
import sys

# ================= CONFIGURAÇÕES =================
PASTA_TESTES = "./musdb18_wav/test"
PASTA_TEMP = "./temp_mixtures"
PASTA_SAIDA = "./estimativas_benchmark_bytesep"
PASTA_REFS_TEMP = "./temp_refs_bytesep"
ARQUIVO_CSV = "resultados_benchmark_bytesep.csv"

# Modelos do ByteSep para avaliar
MODELOS = ["MobileNet_Subbandtime"]
TIPOS_FONTE = ["vocals", "accompaniment"]
# =================================================

def preparar_ambiente():
    """Limpa e cria pastas necessárias."""
    for pasta in [PASTA_TEMP, PASTA_SAIDA, PASTA_REFS_TEMP]:
        if os.path.exists(pasta):
            shutil.rmtree(pasta)
        os.makedirs(pasta, exist_ok=True)

def medir_cpu_ram(processo_pid, picos):
    """Mede o uso de CPU e RAM de um processo em loop enquanto ele estiver vivo."""
    try:
        ps_proc = psutil.Process(processo_pid)
        # psutil cpu_percent precisa ser chamado com um pequeno intervalo para medir corretamente
        while ps_proc.is_running() and ps_proc.status() != psutil.STATUS_ZOMBIE:
            cpu = ps_proc.cpu_percent(interval=0.1)
            ram = ps_proc.memory_info().rss / (1024 * 1024)
            if cpu > picos['cpu']:
                picos['cpu'] = cpu
            if ram > picos['ram']:
                picos['ram'] = ram
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

def executar_bytesep(caminho_mix, pasta_saida_musica, model_type, source_type, picos):
    """Executa o processo CLI do ByteSep para um tipo específico de fonte e rastreia seus recursos."""
    caminho_saida = os.path.join(pasta_saida_musica, f"{source_type}.wav")
    
    comando = [
        sys.executable, "-m", "bytesep", "separate",
        "--model_type", model_type,
        "--source_type", source_type,
        "--audio_path", caminho_mix,
        "--output_path", caminho_saida,
        "--cpu" # Remove flag if GPU is intended, assuming CPU as default for fairness if unknown
    ]
    
    # Remover --cpu se a máquina tiver GPU configurada. Para Mac M1/Intel sem CUDA, --cpu pode ser necessário ou não,
    # O código original bytedance para Spleeter não especificou device. ByteSep usa torch, se não houver CUDA cai para CPU, 
    # mas a CLI tem flag --cpu. Deixarei sem a flag --cpu para usar o padrão do ByteSep (usa cuda se disponivel, senao cpu).
    comando.remove("--cpu")
    
    # processo = subprocess.Popen(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processo = subprocess.Popen(comando, stdout=subprocess.DEVNULL)
    
    # Monitoramento de uso
    medir_cpu_ram(processo.pid, picos)
    processo.wait()

def rodar_benchmark():
    preparar_ambiente()
    
    if not os.path.exists(PASTA_TESTES):
        print(f"Pasta de testes não encontrada: {PASTA_TESTES}")
        return

    musicas = [d for d in os.listdir(PASTA_TESTES) if os.path.isdir(os.path.join(PASTA_TESTES, d))]
    
    if not musicas:
        print("Nenhuma música encontrada na pasta de testes.")
        return

    resultados_finais = []

    for modelo in MODELOS:
        print(f"\n[{modelo}] Iniciando inferência...")
        tempo_total_modelo = 0
        picos = {'cpu': 0, 'ram': 0}
        
        pasta_estimativas_modelo = os.path.join(PASTA_SAIDA, modelo)
        os.makedirs(pasta_estimativas_modelo, exist_ok=True)
        
        for musica in musicas:
            caminho_mix_original = os.path.join(PASTA_TESTES, musica, "mixture.wav")
            pasta_saida_musica = os.path.join(pasta_estimativas_modelo, musica)
            os.makedirs(pasta_saida_musica, exist_ok=True)
            
            print(f"  -> Processando: {musica}")
            inicio_tempo = time.time()
            
            # O ByteSep separa um source por vez pela CLI
            for source_type in TIPOS_FONTE:
                executar_bytesep(caminho_mix_original, pasta_saida_musica, modelo, source_type, picos)
                
            fim_tempo = time.time()
            tempo_total_modelo += (fim_tempo - inicio_tempo)
            
        tempo_medio = tempo_total_modelo / len(musicas)
        
        print(f"[{modelo}] Inferência concluída. Calculando SDR (Qualidade)...")
        
        sdrs_do_modelo = []
        
        # Cria refs temporários para avaliação justa (só os wavs que foram gerados)
        if os.path.exists(PASTA_REFS_TEMP):
            shutil.rmtree(PASTA_REFS_TEMP)
        os.makedirs(PASTA_REFS_TEMP, exist_ok=True)
        
        for musica in musicas:
            ref_dir_track = os.path.join(PASTA_TESTES, musica)
            est_dir_track = os.path.join(pasta_estimativas_modelo, musica)
            ref_isolada = os.path.join(PASTA_REFS_TEMP, musica)
            os.makedirs(ref_isolada, exist_ok=True)
            
            # Isolando as trilhas
            arquivos_estimados = [f for f in os.listdir(est_dir_track) if f.endswith(".wav")]
            
            if not arquivos_estimados:
                print(f"  -> AVISO: Nenhum áudio gerado para {musica}. ByteSep falhou.")
                continue # Pula a avaliação desta música para o código não quebrar

            for inst in arquivos_estimados:
                ref_inst = os.path.join(ref_dir_track, inst)
                if os.path.exists(ref_inst):
                    shutil.copy(ref_inst, os.path.join(ref_isolada, inst))
                else:
                    print(f"  -> AVISO: Arquivo de referência {inst} não encontrado no MUSDB18.")
                        
            try:
                resultados_avaliacao = museval.eval_dir(
                    reference_dir=ref_isolada, 
                    estimates_dir=est_dir_track
                )
                df = resultados_avaliacao.df
                sdr_track = df[df['metric'] == 'SDR']['score'].median()
                sdrs_do_modelo.append(sdr_track)
            except Exception as e:
                print(f"  -> Erro ao avaliar a faixa {musica}: {e}")
                
        sdr_mediano = pd.Series(sdrs_do_modelo).median() if sdrs_do_modelo else 0
        
        resultados_finais.append({
            "Modelo": modelo,
            "Tempo Total (s)": round(tempo_total_modelo, 2),
            "Tempo Médio por Faixa (s)": round(tempo_medio, 2),
            "SDR Mediano (dB)": round(sdr_mediano, 2),
            "Pico CPU (%)": round(picos['cpu'], 2),
            "Pico RAM (MB)": round(picos['ram'], 2)
        })

    # Exportação e Resumo
    print("\n" + "="*80)
    print("RESUMO DO BENCHMARK BYTESEP".center(80))
    print("="*80)
    
    df_resultados = pd.DataFrame(resultados_finais)
    print(df_resultados.to_string(index=False))
    
    df_resultados.to_csv(ARQUIVO_CSV, index=False)
    print(f"\nOs resultados completos foram salvos em '{ARQUIVO_CSV}'.")
    
    # Limpeza
    for pasta in [PASTA_TEMP, PASTA_REFS_TEMP]:
        if os.path.exists(pasta):
            shutil.rmtree(pasta)

if __name__ == "__main__":
    rodar_benchmark()
