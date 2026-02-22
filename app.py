# app.py
from flask import Flask, request, render_template, send_file, jsonify, abort
import os
from werkzeug.utils import secure_filename
from detect_active import process_video  # ← 先ほど提供した関数をインポート

app = Flask(__name__)

# 設定
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 最大200MBまで（調整可）
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# フォルダ作成
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """許可された拡張子かチェック"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """トップページ（アップロードフォーム）"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    """動画アップロード & 処理エンドポイント"""
    if 'file' not in request.files:
        return jsonify({'error': 'ファイルが選択されていません'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'ファイル名が空です'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        # 生成ファイルのパス（上書きされるので固定名でOK。必要ならユーザーIDなどで分離）
        output_funscript = os.path.join(app.config['OUTPUT_FOLDER'], 'generated.funscript')
        output_csv = os.path.join(app.config['OUTPUT_FOLDER'], 'trajectory.csv')
        output_funscript_csv = os.path.join(app.config['OUTPUT_FOLDER'], 'actions.csv')

        try:
            # ここで動画処理を実行
            generated_path = process_video(
                video_path=video_path,
                output_csv=output_csv,
                output_funscript=output_funscript,
                output_funscript_csv=output_funscript_csv,
                # 必要に応じて他のパラメータを追加（例: pos_column='X座標(px)' など）
            )

            # 処理が成功したら、アップロードした元動画を削除（重要！）
            if os.path.exists(video_path):
                os.remove(video_path)

            return jsonify({
                'success': True,
                'message': '処理が完了しました',
                'download_url': '/download'
            })

        except ValueError as ve:
            # 処理側の明示的なエラー（動画が開けないなど）
            if os.path.exists(video_path):
                os.remove(video_path)
            return jsonify({'error': str(ve)}), 400

        except Exception as e:
            # その他の予期せぬエラー
            if os.path.exists(video_path):
                os.remove(video_path)
            import traceback
            traceback.print_exc()  # コンソールに詳細ログ
            return jsonify({'error': f'サーバー内部エラー: {str(e)}'}), 500

    else:
        return jsonify({'error': '対応していないファイル形式です（.mp4, .avi, .mov, .mkv のみ）'}), 400


@app.route('/download')
def download_funscript():
    """Funscriptダウンロード"""
    funscript_path = os.path.join(app.config['OUTPUT_FOLDER'], 'generated.funscript')

    if not os.path.exists(funscript_path):
        abort(404, description="Funscriptファイルが見つかりません。もう一度アップロードしてください。")

    # ダウンロード後、ファイルを削除するかは任意
    # ここでは残す（複数回ダウンロード可能にするため）
    # 削除したい場合は以下を有効に
    # os.remove(funscript_path)  # ← 必要に応じて

    return send_file(
        funscript_path,
        as_attachment=True,
        download_name='generated.funscript',
        mimetype='application/json'
    )


if __name__ == '__main__':
    # デバッグモード（本番ではFalseに）
    app.run(debug=True, host='0.0.0.0', port=5000)
    # 本番では gunicorn などで起動することを推奨
    # 例: gunicorn -w 4 -b 0.0.0.0:5000 app:app