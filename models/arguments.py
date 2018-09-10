
import argparse
parser = argparse.ArgumentParser(description='Multi-task emo2vec')
parser.add_argument('-esize','--esize', help='size of the embedding', required=False, default=100)
parser.add_argument('-id','--id', help='specific id for this exp', required=True, default=None)
parser.add_argument('-l2','--l2', help='l2 regularization', required=False, default=0.1)
parser.add_argument('-dr','--dr', help='dropout regularization', required=False, default=0.0)
parser.add_argument('-emo2vec','--emo2vec', help='pre trained emo2vec', required=False, default=1)
parser.add_argument('-weightedloss','--weightedloss', help='weightedloss', required=False, default=1)
parser.add_argument('-bsz','--bsz', help='Batch_size', required=False, default=32)
parser.add_argument('-lr','--lr', help='Learning Rate', required=False, default=0.001)
parser.add_argument('-max_emo','--max_emo', help='max_emo', required=False, default=1000000)
parser.add_argument('-path','--path', help='path of the file to load', required=False, default=False)
parser.add_argument('-remove_key','--remove_key', help='remove keys', required=False, default="n.a.")
parser.add_argument('-eval', '--eval', help="eval on valid and test", required=False, default=False)
parser.add_argument('-use_glove', '--use_glove', help="use glove embedding for representation ", required=False, default=False)
parser.add_argument('-allow_tuning_emb', '--allow_tuning_emb', help="allow emb to be updated during training", required=False, default=False)

args = vars(parser.parse_args())
print(args)

