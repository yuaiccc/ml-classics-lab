// 适配壳：Cifar10Viz 自带 fetch（/results/cifar10-cnn.json），不依赖播放器帧。
// 这里包一层，吃掉 A 播放器传入的 state/meta props。
import Cifar10Viz from "@/components/Cifar10Viz";

export default function Cifar10Plot() {
  return <Cifar10Viz />;
}
