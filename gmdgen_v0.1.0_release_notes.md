# gmdgen v0.1.0

이 릴리즈는 기존 v0.1.0을 Gemini API 기반 CLI workflow로 덮어쓴다.

* CLI는 `GEMINI_API_KEY` 존재 여부를 확인하되 key 값을 출력하지 않는다.
* `GEMINI_API_KEY`가 설정되어 있으면 live Gemini API smoke check를 수행할 수 있다.
* missing API key는 fake success가 아니라 오류로 처리한다.
* Gemini API가 기본 provider다.
* Ollama/local/qwen은 기본 provider 경로가 아니다.
* OpenAI fallback은 명시 옵션에서만 허용된다.
* GUI는 기본 실행 경로가 아니다.
* CLI가 기본 인터페이스다.
* runtime log는 실시간 출력된다.
* progress는 퍼센트로 표시된다.
* 로그는 category별로 분리된다.
* quality gate 실패 결과는 기본적으로 trusted final artifact가 아니다.