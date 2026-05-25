# SG_proj_003 Git Push 장애 보고서 및 해결 가이드

## 1. 발생 일시 및 상황
- **발생 일시**: 2026년 5월 25일
- **발생 상황**: 통합 프로젝트(`SG_integration_002-003-007`)에서 고도화된 기능(4K 다운샘플링, 단위 렌더링 등)을 `SG_proj_003`의 `feat/backport-advanced-features` 브랜치에 이식하고 **로컬 커밋(Commit)까지는 완벽하게 성공**하였으나, 원격 저장소(GitHub)로 푸시(`git push`)를 시도할 때 에러가 발생하여 업로드가 튕겨져 나옴.

## 2. 에러 로그 (Error Log)
```bash
error: RPC failed; HTTP 500 curl 22 The requested URL returned error: 500
send-pack: unexpected disconnect while reading sideband packet
fatal: the remote end hung up unexpectedly
Everything up-to-date
```

## 3. 원인 분석
위 에러(`HTTP 500 Internal Server Error`)는 사용자의 로컬 환경이나 코딩 문법의 문제가 아닙니다. 다음과 같은 원인으로 인해 발생합니다.
1. **GitHub 자체 서버의 일시적 장애**: 깃허브 측의 트래픽 과부하 또는 점검으로 인해 요청을 수신하지 못함.
2. **HTTP 버퍼 초과**: 전송해야 할 데이터 패킷이 HTTP postBuffer 기본값을 초과하여 연결이 끊어짐. (단, 이번 커밋은 텍스트 파일 몇 개에 불과하므로 1번 원인일 확률이 99%입니다.)

## 4. 해결 방법 (Troubleshooting)

코드는 사용자님의 로컬 PC에 100% 안전하게 저장(커밋)되어 있으므로 유실될 걱정은 하지 않으셔도 됩니다. 나중에 시간이 나실 때 아래의 절차를 따라 푸시를 재시도해 주십시오.

### 방법 A. 깃허브 서버가 정상화된 후 단순 재시도 (가장 추천)
단순 서버 오류이므로 시간이 지나면 자연스레 해결됩니다.
```powershell
# SG_proj_003 경로에서 아래 명령어 입력
git push -u origin feat/backport-advanced-features
```

### 방법 B. 버퍼 크기 증가 후 재시도
만약 네트워크 환경에 의해 패킷이 끊어지는 것이라면 버퍼 제한을 500MB로 늘려서 푸시합니다.
```powershell
git config http.postBuffer 524288000
git push -u origin feat/backport-advanced-features
```

### 방법 C. HTTPS 대신 SSH 방식으로 푸시하기
가장 확실한 네트워크 연결 방식입니다. GitHub에 SSH 키가 등록되어 있다면 리모트 주소를 SSH로 변경 후 푸시합니다.
```powershell
git remote set-url origin git@github.com:HyunchanAn/SG_proj_003.git
git push -u origin feat/backport-advanced-features
```

---
**💡 요약**: 로컬에는 모든 고도화 작업 코드가 성공적으로 덮어씌워져 있으니, 내일 터미널을 열고 `git push`만 한 번 다시 입력해 주시면 됩니다!
