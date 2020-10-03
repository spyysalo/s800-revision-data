#!/usr/bin/env python3

import sys
import os

import nltk

from collections import defaultdict

from nltk.stem import WordNetLemmatizer


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-i', '--ignore-errors', default=False, action='store_true',
                    help='ignore format errors')
    ap.add_argument('nodes', default=None, help='NCBI taxonomy nodes.dmp')
    ap.add_argument('names', default=None, help='NCBI taxonomy names.dmp')
    ap.add_argument('ann', nargs='+', help='annotation')
    return ap


########## annotation ##########


class FormatError(Exception):
    pass


class SpanDeleted(Exception):
    pass


class Annotation(object):
    def __init__(self, id_, type_):
        self.id_ = id_
        self.type_ = type_
        self.normalizations = []
        self.notes = []

    def resolve_references(self, ann_by_id):
        pass

    def add_normalization(self, ann):
        self.normalizations.append(ann)

    def add_note(self, ann):
        self.notes.append(ann)


class Textbound(Annotation):
    def __init__(self, id_, type_, offsets, text):
        Annotation.__init__(self, id_, type_)
        self.text = text
        self.offsets = []
        for start_end in offsets.split(';'):
            start, end = start_end.split()
            self.offsets.append((int(start), int(end)))


class XMLElement(Textbound):
    def __init__(self, id_, type_, offsets, text, attributes):
        Textbound.__init__(self, id_, type_, offsets, text)
        self.attributes = attributes


class ArgAnnotation(Annotation):
    def __init__(self, id_, type_, args):
        Annotation.__init__(self, id_, type_)
        self.args = args

    def resolve_references(self, ann_by_id):
        raise NotImplementedError()    # TODO


class Relation(ArgAnnotation):
    def __init__(self, id_, type_, args):
        ArgAnnotation.__init__(self, id_, type_, args)


class Event(ArgAnnotation):
    def __init__(self, id_, type_, trigger, args):
        ArgAnnotation.__init__(self, id_, type_, args)
        self.trigger = trigger


class Attribute(Annotation):
    def __init__(self, id_, type_, target_id, value):
        Annotation.__init__(self, id_, type_)
        self.target_id = target_id
        self.value = value
        self.target = None

    def resolve_references(self, ann_by_id):
        self.target = ann_by_id[self.target_id]


class Normalization(Annotation):
    def __init__(self, id_, type_, target_id, ref, reftext):
        Annotation.__init__(self, id_, type_)
        self.target_id = target_id
        self.ref = ref
        self.reftext = reftext
        self.target = None

    def resolve_references(self, ann_by_id):
        self.target = ann_by_id[self.target_id]
        self.target.add_normalization(self)


class Equiv(Annotation):
    def __init__(self, id_, type_, targets):
        Annotation.__init__(self, id_, type_)
        self.targets = targets

    def resolve_references(self, ann_by_id):
        raise NotImplementedError()    # TODO


class Note(Annotation):
    def __init__(self, id_, type_, target_id, text):
        Annotation.__init__(self, id_, type_)
        self.target_id = target_id
        self.text = text
        self.target = None

    def resolve_references(self, ann_by_id):
        self.target = ann_by_id[self.target_id]
        self.target.add_note(self)


def parse_xml(fields):
    id_, type_offsets, text, attributes = fields
    type_offsets = type_offsets.split(' ')
    type_, offsets = type_offsets[0], type_offsets[1:]
    return XMLElement(id_, type_, offsets, text, attributes)


def parse_textbound(fields):
    id_, type_offsets, text = fields
    type_offsets = type_offsets.split(' ', 1)
    type_, offsets = type_offsets
    return Textbound(id_, type_, offsets, text)


def parse_relation(fields):
    # allow a variant where the two initial TAB-separated fields are
    # followed by an extra tab
    if len(fields) == 3 and not fields[2]:
        fields = fields[:2]
    id_, type_args = fields
    type_args = type_args.split(' ')
    type_, args = type_args[0], type_args[1:]
    return Relation(id_, type_, args)


def parse_event(fields):
    id_, type_trigger_args = fields
    type_trigger_args = type_trigger_args.split(' ')
    type_trigger, args = type_trigger_args[0], type_trigger_args[1:]
    type_, trigger = type_trigger.split(':')
    return Event(id_, type_, trigger, args)


def parse_attribute(fields):
    id_, type_target_value = fields
    type_target_value = type_target_value.split(' ')
    if len(type_target_value) == 3:
        type_, target, value = type_target_value
    else:
        type_, target = type_target_value
        value = None
    return Attribute(id_, type_, target, value)


def parse_normalization(fields):
    if len(fields) == 3:
        id_, type_target_ref, reftext = fields
    elif len(fields) == 2:    # Allow missing reference text
        id_, type_target_ref = fields
        reftext = ''
    type_, target, ref = type_target_ref.split(' ')
    return Normalization(id_, type_, target, ref, reftext)


def parse_note(fields):
    id_, type_target, text = fields
    type_, target = type_target.split(' ')
    return Note(id_, type_, target, text)


def parse_equiv(fields):
    id_, type_targets = fields
    type_targets = type_targets.split(' ')
    type_, targets = type_targets[0], type_targets[1:]
    return Equiv(id_, type_, targets)


parse_standoff_func = {
    'T': parse_textbound,
    'R': parse_relation,
    'E': parse_event,
    'N': parse_normalization,
    'M': parse_attribute,
    'A': parse_attribute,
    'X': parse_xml,
    '#': parse_note,
    '*': parse_equiv,
}


def parse_standoff_line(l, ln, fn, options):
    try:
        return parse_standoff_func[l[0]](l.split('\t'))
    except Exception:
        if options.ignore_errors:
            error('failed to parse line {} in {}: {}'.format(ln, fn, l))
            return None
        else:
            raise FormatError('error on line {} in {}: {}'.format(ln, fn, l))


def parse_ann_file(fn, options):
    annotations = []
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            if not l or l.isspace():
                continue
            ann = parse_standoff_line(l, ln, fn, options)
            if ann is not None:
                annotations.append(ann)
    return annotations


def resolve_references(annotations, options):
    ann_by_id = {}
    for a in annotations:
        ann_by_id[a.id_] = a
    for a in annotations:
        a.resolve_references(ann_by_id)


########## end annotation ##########


NON_WN_LEMMAS = {
    'bacteria': 'bacterium',
    'fungi': 'fungus',
}


def lemmatize(string):
    if lemmatize.lemmatizer is None:
        nltk.download('wordnet')
        lemmatize.lemmatizer = WordNetLemmatizer()
    if string in NON_WN_LEMMAS:
        return NON_WN_LEMMAS[string]
    else:
        return lemmatize.lemmatizer.lemmatize(string)
lemmatize.lemmatizer = None


def normalize_space(string):
    return ' '.join(string.split())


def output(fn, annotations, ranks, names, options):
    textbounds = [a for a in annotations if isinstance(a, Textbound)]
    textbounds.sort(key=lambda t:(t.offsets[0][0], -t.offsets[-1][1]))
    for t in textbounds:
        refs = [n.ref.replace('Taxonomy:', '') for n in t.normalizations]
        notes = [normalize_space(n.text) for n in t.notes]
        if not refs and t.type_ == 'Out-of-scope':
            continue
        if not refs:
            refs = ['0']
            name_classes = ['NO-NORM']
            norm_ranks = ['NO-NORM']
        else:
            name_classes = names[refs[0]][t.text]
            norm_ranks = [ranks[r] for r in refs]
        if not name_classes:
            name_classes = names[refs[0]][t.text.lower()]
        if not name_classes:
            name_classes = ['UNKNOWN']
        # print('{}\t{}'.format(','.join(refs), t.text))
        fields = []
        fields.append(os.path.splitext(os.path.basename(fn))[0])
        fields.append(','.join(refs))
        fields.append(t.text)
        fields.append(t.type_)
        fields.append(';'.join(notes))
        fields.append(','.join(norm_ranks))
        fields.append(','.join(name_classes))
        print('\t'.join(fields))
        #print('{}\t{}\t{}\t{}'.format(
        #    ','.join(refs), t.text, t.type_, ';'.join(notes)))


def name_variants(text, name_class):
    if name_class == 'scientific name':
        # "Homo sapiens" -> "H. sapiens" etc.
        parts = text.split()
        initial, rest = parts[0][0], ' '.join(parts[1:])
        yield (initial + '. ' + rest, name_class + ' (abbrev)')
        yield (initial + ' ' + rest, name_class + ' (abbrev)')
        yield (initial + '.' + rest, name_class + ' (abbrev)')

    if name_class in ['common name', 'genbank common name', 'blast name']:
        parts = text.split()
        start, last = parts[:-1], parts[-1]
        lemma = lemmatize(last)
        yield (' '.join(start + [lemma]), name_class + ' (lemma)')


def parse_dump_line(line):
    line = line.rstrip('\n')
    fields = line.split('|')
    fields = [f.strip() for f in fields]
    return fields


def read_nodes(fn):
    ranks = {}
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            fields = parse_dump_line(l)
            tax_id, parent_id, rank = fields[:3]
            assert tax_id not in ranks
            ranks[tax_id] = rank
    return ranks


def read_taxnames(fn):
    names = defaultdict(lambda: defaultdict(list))
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            fields = parse_dump_line(l)
            tax_id, text, unique_name, name_class, _ = fields
            names[tax_id][text].append(name_class)
    new_names = defaultdict(lambda: defaultdict(list))
    for tax_id in names:
        for text in names[tax_id]:
            for name_class in names[tax_id][text]:
                for t, c in name_variants(text, name_class):
                    new_names[tax_id][t].append(c)
    for tax_id in new_names:
        for text in new_names[tax_id]:
            for name_class in new_names[tax_id][text]:
                if text not in names[tax_id]:
                    names[tax_id][text].append(name_class)
    return names


def main(argv):
    args = argparser().parse_args(argv[1:])
    ranks = read_nodes(args.nodes)
    names = read_taxnames(args.names)
    for fn in args.ann:
        annotations = parse_ann_file(fn, args)
        resolve_references(annotations, args)
        output(fn, annotations, ranks, names, args)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

