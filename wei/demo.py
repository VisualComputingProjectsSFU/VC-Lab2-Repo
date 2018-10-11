import util.module_util
import ssd_net


net = ssd_net.SsdNet(10)
util.module_util.summary_layers(net, (3, 300, 300))
