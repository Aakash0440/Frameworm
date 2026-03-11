import pathlib

f = pathlib.Path("models/cfg_ddpm.py")
c = f.read_text(encoding="utf-8")
c = c.replace(
    "        self.num_classes = num_classes\n        time_emb_dim = model_channels * 4",
    "        self.num_classes = num_classes\n        self.num_res_blocks = num_res_blocks\n        time_emb_dim = model_channels * 4",
)
f.write_text(c, encoding="utf-8")
print("done")
