/*
 * 손글씨 숫자 인식 - 브라우저(JavaScript) 버전의 핵심 기능입니다.
 * Python 버전(recognizer.py)과 같은 순서로 처리합니다.
 *   1) 획 좌표로 280x280 그림 그리기
 *   2) 살짝 흐리게 하기 (PIL GaussianBlur(1)과 같은 방법)
 *   3) 글씨 부분만 잘라 긴 쪽을 20픽셀로 줄이기 (PIL LANCZOS와 같은 방법)
 *   4) 28x28 가운데에 붙이고 무게중심을 (14, 14)로 옮기기
 *   5) CNN(합성곱 신경망) 계산으로 숫자 0~9의 확률 구하기
 * 가중치는 export_web_model.py가 만든 model.json / model.bin에서 읽습니다.
 * 브라우저에서는 window.손글씨인식, Node.js에서는 require()로 씁니다.
 */
(function (전역) {
  "use strict";

  const 캔버스_크기 = 280;
  const 펜_굵기 = 22;

  // ----- 1) 획 좌표로 그림 그리기 -----

  /** 점 (px, py)에서 선분 (ax, ay)-(bx, by)까지의 거리의 제곱 */
  function 선분_거리_제곱(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay;
    const 길이_제곱 = dx * dx + dy * dy;
    let t = 길이_제곱 > 0 ? ((px - ax) * dx + (py - ay) * dy) / 길이_제곱 : 0;
    t = Math.max(0, Math.min(1, t));
    const qx = ax + t * dx - px, qy = ay + t * dy - py;
    return qx * qx + qy * qy;
  }

  /**
   * 두 점 사이를 펜 굵기만큼 두껍게 칠합니다 (끝은 둥글게).
   * 픽셀 한가운데(x+0.5, y+0.5)가 선에서 r 안쪽이면 칠합니다.
   * r을 펜 굵기의 절반보다 0.25 크게 잡은 것은 PIL로 그린 결과와 가장 비슷해서입니다 (비교 시험으로 정함).
   */
  function 굵은_선_칠하기(그림, ax, ay, bx, by) {
    const r = 펜_굵기 / 2 + 0.25;
    const 왼쪽 = Math.max(0, Math.floor(Math.min(ax, bx) - r - 1));
    const 오른쪽 = Math.min(캔버스_크기 - 1, Math.ceil(Math.max(ax, bx) + r + 1));
    const 위 = Math.max(0, Math.floor(Math.min(ay, by) - r - 1));
    const 아래 = Math.min(캔버스_크기 - 1, Math.ceil(Math.max(ay, by) + r + 1));
    for (let y = 위; y <= 아래; y++) {
      for (let x = 왼쪽; x <= 오른쪽; x++) {
        if (선분_거리_제곱(x + 0.5, y + 0.5, ax, ay, bx, by) <= r * r) 그림[y * 캔버스_크기 + x] = 255;
      }
    }
  }

  /**
   * 획 목록으로 280x280 흑백 그림(검은 바탕 0, 흰 글씨 255)을 만듭니다.
   * 획_목록 예: [[[10, 20], [12, 25], ...], [...]]
   */
  function 획으로_그림_만들기(획_목록) {
    const 그림 = new Uint8Array(캔버스_크기 * 캔버스_크기);
    for (const 획 of 획_목록) {
      if (!획.length) continue;
      굵은_선_칠하기(그림, 획[0][0], 획[0][1], 획[0][0], 획[0][1]);  // 첫 점
      for (let i = 1; i < 획.length; i++) {
        굵은_선_칠하기(그림, 획[i - 1][0], 획[i - 1][1], 획[i][0], 획[i][1]);
      }
    }
    return 그림;
  }

  // ----- 2) 흐리게 하기 (PIL의 GaussianBlur와 같은 '상자 흐림 3번' 방식) -----

  function 흐림_반지름_구하기(반지름, 횟수) {
    const 분산 = (반지름 * 반지름) / 횟수;
    const L = Math.sqrt(12 * 분산 + 1);
    const l = Math.floor((L - 1) / 2);
    let a = (2 * l + 1) * (l * (l + 1) - 3 * 분산);
    a /= 6 * (분산 - (l + 1) * (l + 1));
    return l + a;
  }

  /** 한 줄을 상자 흐림 합니다. 가장자리 밖은 가장자리 값으로 채운 것처럼 계산합니다. */
  function 줄_흐리기(입력, 출력, 길이, 반지름) {
    const 정수_반지름 = Math.floor(반지름);
    const 안쪽_무게 = 1 / (반지름 * 2 + 1);
    const 바깥_무게 = (1 - (정수_반지름 * 2 + 1) * 안쪽_무게) / 2;
    const 값 = (i) => 입력[Math.max(0, Math.min(길이 - 1, i))];
    for (let x = 0; x < 길이; x++) {
      let 합 = 0;
      for (let k = -정수_반지름; k <= 정수_반지름; k++) 합 += 값(x + k);
      합 = 합 * 안쪽_무게 + (값(x - 정수_반지름 - 1) + 값(x + 정수_반지름 + 1)) * 바깥_무게;
      출력[x] = Math.min(255, Math.floor(합 + 0.5));
    }
  }

  function 흐리게_하기(그림, 가로, 세로, 반지름 = 1, 횟수 = 3) {
    const 상자_반지름 = 흐림_반지름_구하기(반지름, 횟수);
    let 현재 = Uint8Array.from(그림);
    // 가로 방향으로 3번
    const 줄 = new Uint8Array(가로), 결과줄 = new Uint8Array(가로);
    for (let y = 0; y < 세로; y++) {
      for (let x = 0; x < 가로; x++) 줄[x] = 현재[y * 가로 + x];
      for (let n = 0; n < 횟수; n++) {
        줄_흐리기(줄, 결과줄, 가로, 상자_반지름);
        줄.set(결과줄);
      }
      for (let x = 0; x < 가로; x++) 현재[y * 가로 + x] = 줄[x];
    }
    // 세로 방향으로 3번
    const 칸 = new Uint8Array(세로), 결과칸 = new Uint8Array(세로);
    for (let x = 0; x < 가로; x++) {
      for (let y = 0; y < 세로; y++) 칸[y] = 현재[y * 가로 + x];
      for (let n = 0; n < 횟수; n++) {
        줄_흐리기(칸, 결과칸, 세로, 상자_반지름);
        칸.set(결과칸);
      }
      for (let y = 0; y < 세로; y++) 현재[y * 가로 + x] = 칸[y];
    }
    return 현재;
  }

  // ----- 3) 크기 줄이기 (PIL의 LANCZOS와 같은 방법) -----

  function 사인크(x) {
    if (x === 0) return 1;
    const px = Math.PI * x;
    return Math.sin(px) / px;
  }

  function 란초스(x) {
    return -3 < x && x < 3 ? 사인크(x) * 사인크(x / 3) : 0;
  }

  /** 길이 입력_길이 → 출력_길이로 줄일 때, 출력 칸마다 어떤 입력 칸을 얼마나 섞을지 계산합니다. */
  function 섞을_비율_구하기(입력_길이, 출력_길이) {
    const 배율 = 입력_길이 / 출력_길이;
    const 필터_배율 = Math.max(배율, 1);
    const 범위 = 3 * 필터_배율;
    const 목록 = [];
    for (let 출력 = 0; 출력 < 출력_길이; 출력++) {
      const 중심 = (출력 + 0.5) * 배율;
      const 시작 = Math.max(Math.trunc(중심 - 범위 + 0.5), 0);
      const 끝 = Math.min(Math.trunc(중심 + 범위 + 0.5), 입력_길이);
      let 비율들 = [];
      let 합 = 0;
      for (let x = 시작; x < 끝; x++) {
        const w = 란초스((x - 중심 + 0.5) / 필터_배율);
        비율들.push(w);
        합 += w;
      }
      // PIL처럼 정수(22비트 고정소수점)로 바꿉니다.
      비율들 = 비율들.map((w) => {
        const v = 합 !== 0 ? w / 합 : 0;
        return Math.trunc(v * (1 << 22) + (v < 0 ? -0.5 : 0.5));
      });
      목록.push({ 시작, 비율들 });
    }
    return 목록;
  }

  function 한_방향_줄이기(그림, 가로, 세로, 새_길이, 가로_방향) {
    const 입력_길이 = 가로_방향 ? 가로 : 세로;
    const 비율표 = 섞을_비율_구하기(입력_길이, 새_길이);
    const 새_가로 = 가로_방향 ? 새_길이 : 가로;
    const 새_세로 = 가로_방향 ? 세로 : 새_길이;
    const 결과 = new Uint8Array(새_가로 * 새_세로);
    for (let y = 0; y < 새_세로; y++) {
      for (let x = 0; x < 새_가로; x++) {
        const { 시작, 비율들 } = 비율표[가로_방향 ? x : y];
        let 합 = 1 << 21;  // 반올림용
        for (let k = 0; k < 비율들.length; k++) {
          const 값 = 가로_방향 ? 그림[y * 가로 + 시작 + k] : 그림[(시작 + k) * 가로 + x];
          합 += 값 * 비율들[k];
        }
        결과[y * 새_가로 + x] = Math.max(0, Math.min(255, Math.floor(합 / (1 << 22))));
      }
    }
    return 결과;
  }

  function 크기_바꾸기(그림, 가로, 세로, 새_가로, 새_세로) {
    let 현재 = 그림, 현재_가로 = 가로;
    if (새_가로 !== 가로) {
      현재 = 한_방향_줄이기(현재, 가로, 세로, 새_가로, true);
      현재_가로 = 새_가로;
    }
    if (새_세로 !== 세로) 현재 = 한_방향_줄이기(현재, 현재_가로, 세로, 새_세로, false);
    return 현재;
  }

  /** Python의 round()처럼 0.5는 가까운 짝수로 반올림합니다. */
  function 짝수_반올림(x) {
    const 내림 = Math.floor(x);
    const 차이 = x - 내림;
    if (차이 > 0.5) return 내림 + 1;
    if (차이 < 0.5) return 내림;
    return 내림 % 2 === 0 ? 내림 : 내림 + 1;
  }

  // ----- 4) MNIST 형식(28x28, 가운데 정렬)으로 바꾸기 -----

  function mnist_형식으로_변환(그림, 가로 = 캔버스_크기, 세로 = 캔버스_크기) {
    // 1) 글씨가 있는 부분 찾기
    let 위 = 세로, 아래 = -1, 왼쪽 = 가로, 오른쪽 = -1;
    for (let y = 0; y < 세로; y++) {
      for (let x = 0; x < 가로; x++) {
        if (그림[y * 가로 + x] > 0) {
          if (y < 위) 위 = y;
          if (y > 아래) 아래 = y;
          if (x < 왼쪽) 왼쪽 = x;
          if (x > 오른쪽) 오른쪽 = x;
        }
      }
    }
    if (아래 < 0) return null;  // 아무것도 안 그렸으면 인식하지 않습니다.

    const 잘린_가로 = 오른쪽 - 왼쪽 + 1, 잘린_세로 = 아래 - 위 + 1;
    const 잘린_그림 = new Uint8Array(잘린_가로 * 잘린_세로);
    for (let y = 0; y < 잘린_세로; y++) {
      for (let x = 0; x < 잘린_가로; x++) {
        잘린_그림[y * 잘린_가로 + x] = 그림[(위 + y) * 가로 + (왼쪽 + x)];
      }
    }

    // 2) 긴 쪽을 20픽셀로 줄이기
    const 배율 = 20 / Math.max(잘린_가로, 잘린_세로);
    const 새_가로 = Math.max(1, 짝수_반올림(잘린_가로 * 배율));
    const 새_세로 = Math.max(1, 짝수_반올림(잘린_세로 * 배율));
    const 작은_그림 = 크기_바꾸기(잘린_그림, 잘린_가로, 잘린_세로, 새_가로, 새_세로);

    // 3) 28x28 가운데에 붙이기
    const 붙인_그림 = new Uint8Array(28 * 28);
    const 붙일_x = Math.floor((28 - 새_가로) / 2), 붙일_y = Math.floor((28 - 새_세로) / 2);
    for (let y = 0; y < 새_세로; y++) {
      for (let x = 0; x < 새_가로; x++) {
        붙인_그림[(붙일_y + y) * 28 + (붙일_x + x)] = 작은_그림[y * 새_가로 + x];
      }
    }

    // 4) 무게중심을 (14, 14)로 옮기기
    let 전체 = 0, 세로_합 = 0, 가로_합 = 0;
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const v = 붙인_그림[y * 28 + x];
        전체 += v; 세로_합 += y * v; 가로_합 += x * v;
      }
    }
    const 이동_가로 = 짝수_반올림(14 - 가로_합 / 전체);
    const 이동_세로 = 짝수_반올림(14 - 세로_합 / 전체);
    const 결과 = new Uint8Array(28 * 28);
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const 원래_x = x - 이동_가로, 원래_y = y - 이동_세로;
        if (원래_x >= 0 && 원래_x < 28 && 원래_y >= 0 && 원래_y < 28) {
          결과[y * 28 + x] = 붙인_그림[원래_y * 28 + 원래_x];
        }
      }
    }
    return 결과;
  }

  // ----- 5) CNN 계산 (model.py의 DigitCNN과 같은 구조) -----

  /** model.json(정보)과 model.bin(숫자들)으로 모델을 만듭니다. */
  function 모델_만들기(정보, 이진_데이터) {
    const 숫자들 = new Float32Array(이진_데이터);
    const 텐서 = (이름) => {
      const { shape, offset } = 정보.tensors[이름];
      const 개수 = shape.reduce((a, b) => a * b, 1);
      return 숫자들.subarray(offset, offset + 개수);
    };
    return {
      conv1: { w: 텐서("conv1.weight"), b: 텐서("conv1.bias"), 입력: 1, 출력: 32 },
      conv2: { w: 텐서("conv2.weight"), b: 텐서("conv2.bias"), 입력: 32, 출력: 64 },
      fc1: { w: 텐서("fc1.weight"), b: 텐서("fc1.bias"), 입력: 3136, 출력: 128 },
      fc2: { w: 텐서("fc2.weight"), b: 텐서("fc2.bias"), 입력: 128, 출력: 10 },
    };
  }

  /** 3x3 합성곱(가장자리 0으로 채움) + ReLU */
  function 합성곱_렐루(입력, 크기, 층) {
    const 출력 = new Float32Array(층.출력 * 크기 * 크기);
    for (let o = 0; o < 층.출력; o++) {
      for (let y = 0; y < 크기; y++) {
        for (let x = 0; x < 크기; x++) {
          let 합 = 층.b[o];
          for (let i = 0; i < 층.입력; i++) {
            for (let ky = 0; ky < 3; ky++) {
              const yy = y + ky - 1;
              if (yy < 0 || yy >= 크기) continue;
              for (let kx = 0; kx < 3; kx++) {
                const xx = x + kx - 1;
                if (xx < 0 || xx >= 크기) continue;
                합 += 입력[(i * 크기 + yy) * 크기 + xx] * 층.w[((o * 층.입력 + i) * 3 + ky) * 3 + kx];
              }
            }
          }
          출력[(o * 크기 + y) * 크기 + x] = 합 > 0 ? 합 : 0;
        }
      }
    }
    return 출력;
  }

  /** 2x2 최대 풀링 (크기를 절반으로) */
  function 최대_풀링(입력, 채널, 크기) {
    const 반 = 크기 / 2;
    const 출력 = new Float32Array(채널 * 반 * 반);
    for (let c = 0; c < 채널; c++) {
      for (let y = 0; y < 반; y++) {
        for (let x = 0; x < 반; x++) {
          const 기준 = (c * 크기 + y * 2) * 크기 + x * 2;
          출력[(c * 반 + y) * 반 + x] = Math.max(
            입력[기준], 입력[기준 + 1], 입력[기준 + 크기], 입력[기준 + 크기 + 1]);
        }
      }
    }
    return 출력;
  }

  /** 완전연결층 (렐루 여부 선택) */
  function 완전연결(입력, 층, 렐루) {
    const 출력 = new Float32Array(층.출력);
    for (let o = 0; o < 층.출력; o++) {
      let 합 = 층.b[o];
      const 기준 = o * 층.입력;
      for (let i = 0; i < 층.입력; i++) 합 += 입력[i] * 층.w[기준 + i];
      출력[o] = 렐루 && 합 < 0 ? 0 : 합;
    }
    return 출력;
  }

  function 소프트맥스(점수) {
    const 최대 = Math.max(...점수);
    const 지수 = Array.from(점수, (v) => Math.exp(v - 최대));
    const 합 = 지수.reduce((a, b) => a + b, 0);
    return 지수.map((v) => v / 합);
  }

  /** 28x28 그림(0~255)을 넣으면 숫자 0~9의 확률 10개를 돌려줍니다. */
  function 모델_계산(모델, 작은_그림) {
    let x = Float32Array.from(작은_그림, (v) => v / 255);
    x = 최대_풀링(합성곱_렐루(x, 28, 모델.conv1), 32, 28);   // 32 x 14 x 14
    x = 최대_풀링(합성곱_렐루(x, 14, 모델.conv2), 64, 14);   // 64 x 7 x 7
    x = 완전연결(x, 모델.fc1, true);                          // 128
    return 소프트맥스(완전연결(x, 모델.fc2, false));          // 10
  }

  /**
   * 280x280 그림을 받아 숫자를 알아맞힙니다.
   * 돌려주는 값: { 작은_그림: 28x28 Uint8Array, 확률: 10개 배열 } (아무것도 안 그렸으면 null)
   */
  function 숫자_인식하기(모델, 그림) {
    const 작은_그림 = mnist_형식으로_변환(흐리게_하기(그림, 캔버스_크기, 캔버스_크기));
    if (!작은_그림) return null;
    return { 작은_그림, 확률: 모델_계산(모델, 작은_그림) };
  }

  /** 브라우저에서 model.json, model.bin을 내려받아 모델을 준비합니다. */
  async function 모델_불러오기(기준_주소 = "") {
    const [정보, 이진] = await Promise.all([
      fetch(기준_주소 + "model.json").then((r) => r.json()),
      fetch(기준_주소 + "model.bin").then((r) => r.arrayBuffer()),
    ]);
    return 모델_만들기(정보, 이진);
  }

  const 내보낼_것 = {
    캔버스_크기, 펜_굵기,
    획으로_그림_만들기, 흐리게_하기, mnist_형식으로_변환,
    모델_만들기, 모델_불러오기, 모델_계산, 숫자_인식하기,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = 내보낼_것;
  else 전역.손글씨인식 = 내보낼_것;
})(typeof window !== "undefined" ? window : globalThis);
